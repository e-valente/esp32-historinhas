"""Prepare the pinned Historinhas corpus for PLE training.

The source is a 1.6 GB JSON array, so this module never loads it into memory or
cuts it at an arbitrary byte boundary. It verifies the immutable Hugging Face
revision, streams documents from ``texto``, and writes uint16 token bins. The
manifest beside each vocabulary records every input needed to audit a run.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw" / "historinhas_train.json"
DATASET = "Boakpe/historinhas"
REVISION = "5c9e42e00b6dc440a15c06b76dcefe506ba6c7fb"
SOURCE_SHA256 = "bca0389cc8644071701e4793e232f120c4b98b17545a119bb93c52a21262cf3c"
SOURCE_BYTES = 1_634_706_508
URL = (f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/"
       "historinhas_train.json")
TEXT_FIELD = "texto"
DEFAULT_VOCAB = 32768
VAL_FRACTION = 0.005
TOKENIZER_TRAIN_BYTES = 40 * 1024 * 1024
EOT = "<|endoftext|>"


def variant_dir(vocab_size):
    return DATA / f"vocab-{vocab_size}"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_is_valid():
    return (RAW.is_file() and RAW.stat().st_size == SOURCE_BYTES and
            sha256_file(RAW) == SOURCE_SHA256)


def download():
    """Fetch the exact LFS object atomically and verify it before use."""
    if source_is_valid():
        print(f"already have verified {RAW}")
        return
    if RAW.exists():
        print(f"removing unverified source {RAW}")
        RAW.unlink()

    RAW.parent.mkdir(parents=True, exist_ok=True)
    temporary = RAW.with_suffix(".json.partial")
    temporary.unlink(missing_ok=True)
    digest, received = hashlib.sha256(), 0
    print(f"downloading {DATASET}@{REVISION[:12]} ({SOURCE_BYTES / 1e9:.2f} GB)...")
    try:
        with requests.get(URL, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with open(temporary, "wb") as output:
                for block in response.iter_content(chunk_size=1 << 20):
                    if not block:
                        continue
                    output.write(block)
                    digest.update(block)
                    received += len(block)
                    if received % (100 << 20) < len(block):
                        print(f"  {received / 1e9:.2f} GB", flush=True)
        got = digest.hexdigest()
        if received != SOURCE_BYTES or got != SOURCE_SHA256:
            raise SystemExit(
                f"source integrity error: got {received} bytes / {got}, expected "
                f"{SOURCE_BYTES} bytes / {SOURCE_SHA256}")
        os.replace(temporary, RAW)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"verified {RAW}")


def normalize(text):
    """Keep corpus spelling intact while making line endings platform-independent."""
    if not isinstance(text, str):
        raise SystemExit(f"{TEXT_FIELD!r} must be a string, got {type(text).__name__}")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def is_validation(text):
    """Stable document-level split; duplicate text always stays in one split."""
    value = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    return value < int(VAL_FRACTION * (1 << 64))


def documents(split=None, max_docs=None):
    """Yield normalized non-empty texts, optionally from one deterministic split."""
    emitted = 0
    for row_number, row in enumerate(json_array_items(RAW)):
            if not isinstance(row, dict) or TEXT_FIELD not in row:
                raise SystemExit(f"row {row_number} does not contain {TEXT_FIELD!r}")
            text = normalize(row[TEXT_FIELD])
            if not text.strip():
                continue
            val = is_validation(text)
            if split == "train" and val:
                continue
            if split == "val" and not val:
                continue
            yield text
            emitted += 1
            if max_docs is not None and emitted >= max_docs:
                return


def json_array_items(path):
    """Yield a top-level JSON array using the standard library and bounded RAM."""
    decoder = json.JSONDecoder()
    buffer, first, finished = "", True, False
    with open(path, "r", encoding="utf-8") as source:
        while not finished:
            if len(buffer) < 65536:
                buffer += source.read(1 << 20)
            buffer = buffer.lstrip()
            if not buffer:
                more = source.read(1 << 20)
                if not more:
                    break
                buffer += more
                continue
            if first:
                if buffer[0] != "[":
                    raise SystemExit("Historinhas source must be a top-level JSON array")
                buffer, first = buffer[1:], False
                continue
            if buffer[0] == "]":
                finished = True
                continue
            if buffer[0] == ",":
                buffer = buffer[1:].lstrip()
            try:
                item, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                more = source.read(1 << 20)
                if not more:
                    raise SystemExit("invalid or truncated Historinhas JSON")
                buffer += more
                continue
            yield item
            buffer = buffer[end:]
    if not finished:
        raise SystemExit("invalid or truncated Historinhas JSON: missing closing ]")


def tokenizer_training_text(max_docs):
    """Feed a deterministic <=40 MiB train-only sample without retaining it."""
    used = 0
    for text in documents("train", max_docs):
        size = len(text.encode("utf-8"))
        if used and used + size > TOKENIZER_TRAIN_BYTES:
            return
        used += size
        yield text


def train_tokenizer(out, vocab_size, max_docs):
    path = out / "tokenizer.json"
    if path.exists():
        print(f"reusing {path}")
        return Tokenizer.from_file(str(path))

    print(f"training UTF-8 ByteLevel BPE without regex, vocab={vocab_size}...")
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    # One input string is one BPE pre-token. This avoids embedding a Unicode
    # regex engine in the ESP32 and is exactly mirrored by bpe_tokenizer.h.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(tokenizer_training_text(max_docs), trainer=trainer)
    tokenizer.save(str(path))
    return tokenizer


def encode_split(tokenizer, split, out_path, max_docs):
    """Stream one document split to an atomic uint16 bin."""
    eot = tokenizer.token_to_id(EOT)
    if eot is None:
        raise SystemExit(f"tokenizer has no {EOT!r} token")
    vocabulary = tokenizer.get_vocab_size()
    temporary = out_path.with_suffix(".bin.partial")
    temporary.unlink(missing_ok=True)
    docs_seen = token_count = text_bytes = 0
    batch = []
    with open(temporary, "wb") as output:
        for text in documents(split, max_docs):
            batch.append(text)
            if len(batch) < 256:
                continue
            docs, tokens, raw_bytes = encode_batch(tokenizer, batch, eot, vocabulary, output)
            docs_seen += docs
            token_count += tokens
            text_bytes += raw_bytes
            batch.clear()
        if batch:
            docs, tokens, raw_bytes = encode_batch(tokenizer, batch, eot, vocabulary, output)
            docs_seen += docs
            token_count += tokens
            text_bytes += raw_bytes
    if docs_seen == 0:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"no documents selected for {split}")
    os.replace(temporary, out_path)
    print(f"{split}: {docs_seen:,} documents, {token_count:,} tokens")
    return {"documents": docs_seen, "tokens": token_count, "text_bytes": text_bytes}


def encode_batch(tokenizer, texts, eot, vocabulary, output):
    ids, text_bytes = [], 0
    for text, encoded in zip(texts, tokenizer.encode_batch(texts)):
        if encoded.ids and max(encoded.ids) >= vocabulary:
            raise SystemExit("tokenizer emitted an id outside its vocabulary")
        ids.extend(encoded.ids)
        ids.append(eot)
        text_bytes += len(text.encode("utf-8"))
    np.asarray(ids, dtype=np.uint16).tofile(output)
    return len(texts), len(ids), text_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--max-docs", type=int, default=None,
                        help="development-only deterministic cap per split")
    args = parser.parse_args()
    if not 0 < args.vocab <= 65536:
        raise SystemExit("--vocab must be 1..65536 because token bins are uint16")
    if args.max_docs is not None and args.max_docs <= 0:
        raise SystemExit("--max-docs must be positive")

    out = variant_dir(args.vocab)
    out.mkdir(parents=True, exist_ok=True)
    download()
    tokenizer = train_tokenizer(out, args.vocab, args.max_docs)
    train = encode_split(tokenizer, "train", out / "train.bin", args.max_docs)
    val = encode_split(tokenizer, "val", out / "val.bin", args.max_docs)
    tokenizer_sha = sha256_file(out / "tokenizer.json")
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "license": "Apache-2.0",
        "source": {"path": "historinhas_train.json", "bytes": SOURCE_BYTES,
                   "sha256": SOURCE_SHA256},
        "text_field": TEXT_FIELD,
        "normalization": "CRLF and CR normalized to LF; all other UTF-8 preserved",
        "split": {"method": "sha256(normalized texto) first 64 bits", "val_fraction": VAL_FRACTION},
        "tokenizer": {"type": "ByteLevel BPE", "use_regex": False,
                      "vocab_requested": args.vocab,
                      "vocab_active": tokenizer.get_vocab_size(),
                      "train_bytes_limit": TOKENIZER_TRAIN_BYTES,
                      "sha256": tokenizer_sha},
        "max_docs_per_split": args.max_docs,
        "train": train,
        "val": val,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    total_bytes = train["text_bytes"] + val["text_bytes"]
    total_tokens = train["tokens"] + val["tokens"]
    print(f"manifest: {out / 'manifest.json'}")
    print(f"compression: {total_bytes / total_tokens:.2f} UTF-8 bytes/token")


if __name__ == "__main__":
    sys.exit(main())
