"""Generate text from a trained run, so the loss numbers have something behind them.

The tokenizer is required and is checked against the hash the run recorded.
Checkpoints without that hash can only produce a warning. Decoding a
32,768-vocabulary run with a 4,096-tokenizer produces plausible nonsense rather
than an error, which is why the check exists.
"""

import argparse
import hashlib
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import Config, TinyLM

from .train import get_device

def verify_tokenizer(ck, tokenizer_path):
    """Refuse a tokenizer that did not train this checkpoint.

    The size check alone is not enough: a smaller wrong tokenizer passes it and
    silently produces a model with the wrong output vocabulary. Checkpoints
    written before this was recorded carry no hash, so they can only be warned
    about. Sampling is exploratory, so a warning is enough here; the exporter
    refuses instead, because it produces a distributable binary.
    """
    want = ck.get("tokenizer_sha256")
    got = hashlib.sha256(open(tokenizer_path, "rb").read()).hexdigest()
    if want is None:
        print(f"WARNING: checkpoint records no tokenizer hash; cannot verify "
              f"{os.path.basename(tokenizer_path)} is the right one")
        return
    if want != got:
        raise SystemExit(
            f"tokenizer mismatch: {tokenizer_path} is sha256 {got[:16]}... but "
            f"this checkpoint was trained with {want[:16]}...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="path to a runs/*.pt checkpoint")
    ap.add_argument("--tokenizer", required=True,
                    help="tokenizer this run was trained with, e.g. "
                         "data/vocab-32768/tokenizer.json")
    ap.add_argument("--prompt", default="Era uma vez")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40,
                    help="sample only from this many likely tokens; zero disables it")
    ap.add_argument("--min-new-tokens", type=int, default=24,
                    help="do not accept <|endoftext|> before this many new tokens; use 0 for faithful EOT evaluation")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed; generation is stochastic at temperature > 0")
    args = ap.parse_args()
    if args.tokens < 0:
        ap.error("--tokens must not be negative")
    if args.temperature <= 0:
        ap.error("--temperature must be positive")
    if args.top_k < 0:
        ap.error("--top-k must be zero or positive")
    if not 0 <= args.min_new_tokens <= args.tokens:
        ap.error("--min-new-tokens must be between zero and --tokens")

    torch.manual_seed(args.seed)
    device = get_device()
    ck = torch.load(args.run, map_location=device, weights_only=False)
    model = TinyLM(Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["state"])
    model.eval()

    verify_tokenizer(ck, args.tokenizer)
    tok = Tokenizer.from_file(args.tokenizer)
    eot = tok.token_to_id("<|endoftext|>")
    if eot is None:
        raise SystemExit("tokenizer has no <|endoftext|> token; cannot stop between stories")
    ids = torch.tensor([tok.encode(args.prompt).ids], device=device)

    b = model.param_budget()
    print(f"{Path(args.run).name}  core={b['core']:,} table={b['table']:,}\n")
    for i in range(args.samples):
        out = model.generate(ids, args.tokens, temperature=args.temperature,
                             top_k=args.top_k, eos_token_id=eot,
                             min_new_tokens=args.min_new_tokens)
        new_ids = out[0, ids.size(1):].tolist()
        print(f"--- sample {i + 1} ---")
        print(tok.decode(out[0].tolist()).strip(), "\n")
        if eot in new_ids:
            print(f"[EOT após {new_ids.index(eot)} tokens emitidos]\n")
        else:
            print(f"[limite de {args.tokens} tokens atingido]\n")


if __name__ == "__main__":
    main()
