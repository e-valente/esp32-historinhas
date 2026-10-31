"""Train one ablation arm and report val loss at matched core-parameter budget."""

import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import torch

from model import Config, TinyLM, make_model

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# A vocabulary variant keeps its tokenizer and token bins together.
DATA = ROOT / "data"
RUNS = str(ROOT / "runs")
RESUME_VERSION = 1


def variant_dir(vocab_size):
    return DATA / f"vocab-{vocab_size}"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Batcher:
    def __init__(self, split, batch_size, seq_len, device, dataset, seed=0):
        # The bins are uint16. Reading a wider vocabulary through that dtype
        # yields plausible token ids rather than an error, so check before
        # opening.
        self.data = np.memmap(dataset / f"{split}.bin", dtype=np.uint16, mode="r")
        if len(self.data) <= seq_len + 1:
            raise SystemExit(
                f"{dataset / f'{split}.bin'} has {len(self.data)} tokens, "
                f"but --seq-len {seq_len} needs more")
        self.bs, self.sl, self.device = batch_size, seq_len, device
        # Batch order is part of the run. Without the seed here, torch.manual_seed
        # fixes initialisation only and two runs at the same --seed still see
        # different data order. Validation keeps a fixed stream so every arm is
        # scored on identical batches.
        self.rng = np.random.default_rng(1234 if split == "val" else seed)

    def __call__(self):
        ix = self.rng.integers(0, len(self.data) - self.sl - 1, self.bs)
        x = np.stack([self.data[i : i + self.sl] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1 : i + 1 + self.sl] for i in ix]).astype(np.int64)
        return torch.from_numpy(x).to(self.device), torch.from_numpy(y).to(self.device)


@torch.no_grad()
def evaluate(model, batcher, iters):
    model.eval()
    batcher.rng = np.random.default_rng(1234)  # same val batches for every arm
    losses = [model(*batcher())[1].item() for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)


def lr_at(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 * peak + 0.9 * peak * 0.5 * (1 + math.cos(math.pi * p))


def run_name(args):
    return f"{args.arm}{'-' + args.tag if args.tag else ''}-s{args.seed}"


def training_config(args):
    """Fields that must not change when restoring optimizer state."""
    return {
        "batch_size": args.batch_size,
        "steps": args.steps,
        "lr": args.lr,
        "warmup": args.warmup,
        "eval_every": args.eval_every,
        "eval_iters": args.eval_iters,
        "target_core": args.target_core,
        "fixed_ffn": args.fixed_ffn,
        "seed": args.seed,
        "save_every": args.save_every,
    }


def capture_rng(train_batcher, device):
    state = {
        "torch_cpu": torch.get_rng_state(),
        "numpy_global": np.random.get_state(),
        "train_batcher": train_batcher.rng.bit_generator.state,
    }
    if device == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, train_batcher, device):
    required = {"torch_cpu", "numpy_global", "train_batcher"}
    missing = required - set(state)
    if missing:
        raise SystemExit("resume checkpoint has incomplete RNG state: "
                         + ", ".join(sorted(missing)))
    torch.set_rng_state(state["torch_cpu"])
    np.random.set_state(state["numpy_global"])
    train_batcher.rng.bit_generator.state = state["train_batcher"]
    if device == "cuda":
        if "torch_cuda" not in state:
            raise SystemExit("resume checkpoint has no CUDA RNG state")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def resume_checkpoint(model, opt, cfg, args, name, tok_sha, data_manifest_sha,
                      train_batcher, device):
    """Load a crash checkpoint and reject anything that changes its trajectory."""
    ck = torch.load(args.resume, map_location=device, weights_only=False)
    resume = ck.get("resume")
    if not isinstance(resume, dict):
        raise SystemExit(
            f"{args.resume} cannot resume training: it has no optimizer/RNG state. "
            "Checkpoints written before --resume support can still be sampled or exported.")
    if resume.get("version") != RESUME_VERSION:
        raise SystemExit(f"{args.resume} has unsupported resume version "
                         f"{resume.get('version')!r}")
    if ck.get("name") != name:
        raise SystemExit(f"resume name mismatch: {args.resume} is {ck.get('name')!r}, "
                         f"but this command names {name!r}")
    if ck.get("cfg") != cfg.__dict__:
        raise SystemExit("resume model configuration differs from this command")
    if ck.get("tokenizer_sha256") != tok_sha:
        raise SystemExit("resume tokenizer hash differs from the prepared data")
    if ck.get("data_manifest_sha256") != data_manifest_sha:
        raise SystemExit("resume data manifest hash differs from the prepared data")
    expected_training = training_config(args)
    if ck.get("training") != expected_training:
        raise SystemExit(
            "resume training configuration differs; keep batch size, optimizer "
            "schedule, evaluation cadence, target and total --steps unchanged")
    next_step = resume.get("next_step")
    if not isinstance(next_step, int) or not 0 <= next_step <= args.steps:
        raise SystemExit(f"resume next_step is invalid for --steps {args.steps}: {next_step!r}")
    history, best, elapsed = resume.get("history"), resume.get("best_val"), resume.get("wall_seconds")
    if not isinstance(history, list) or not isinstance(best, (int, float)) or not isinstance(elapsed, (int, float)):
        raise SystemExit("resume checkpoint has incomplete progress state")
    if "optimizer" not in resume or "rng" not in resume:
        raise SystemExit("resume checkpoint has no optimizer or RNG state")
    model.load_state_dict(ck["state"])
    opt.load_state_dict(resume["optimizer"])
    restore_rng(resume["rng"], train_batcher, device)
    return next_step, history, best, float(elapsed)


def checkpoint_payload(model, opt, cfg, args, name, tok_sha, data_manifest_sha,
                       manifest, next_step, history, best, wall_seconds,
                       train_batcher, device):
    return {
        "cfg": cfg.__dict__, "state": model.state_dict(),
        "tokenizer_sha256": tok_sha,
        "data_manifest_sha256": data_manifest_sha,
        "dataset": manifest.get("dataset"),
        "dataset_revision": manifest.get("revision"),
        "seed": args.seed, "tag": args.tag, "name": name,
        "training": training_config(args),
        "resume": {
            "version": RESUME_VERSION,
            "next_step": next_step,
            "history": history,
            "best_val": best,
            "wall_seconds": wall_seconds,
            "optimizer": opt.state_dict(),
            "rng": capture_rng(train_batcher, device),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        required=True,
        choices=["baseline", "ple", "ple_notable", "fatembed", "bigcore"],
    )
    ap.add_argument("--target-core", type=int, default=1_500_000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=40)
    ap.add_argument("--ple-dim", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--fixed-ffn", type=int, default=None,
                    help="pin ffn_hidden and skip the core solver (table-scaling sweep)")
    # Published experiments always pass --vocab; the default is for ad-hoc runs.
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", default=None, metavar="CHECKPOINT",
                    help="resume a matching .pt or .resume.pt checkpoint")
    ap.add_argument("--save-every", type=int, default=250,
                    help="write runs/<name>.resume.pt every N completed steps")
    args = ap.parse_args()

    # Before anything expensive: the tokenizer that produced these bins. Its
    # hash is what lets the exporter and sampler refuse a mismatched tokenizer
    # later, and a checkpoint written without it cannot be tied to one. Failing
    # here costs nothing; failing after a 30-minute run costs the run.
    if not 0 < args.vocab <= 65536:
        raise SystemExit(f"--vocab must be 1..65536; the token bins are uint16 "
                          f"and are memmapped as uint16")
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.save_every <= 0:
        raise SystemExit("--save-every must be positive")

    dataset = variant_dir(args.vocab)
    tok_path = dataset / "tokenizer.json"
    manifest_path = dataset / "manifest.json"
    if not os.path.exists(tok_path):
        raise SystemExit(
            f"{tok_path} missing. The bins this run trains on came from it, and "
            f"without its hash the checkpoint cannot be tied to a tokenizer. "
            f"Run: python -m research.prepare --vocab {args.vocab}")
    tok_sha = hashlib.sha256(tok_path.read_bytes()).hexdigest()
    if not manifest_path.is_file():
        raise SystemExit(
            f"{manifest_path} missing; re-run research.prepare so "
            "the checkpoint can record its corpus provenance")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("tokenizer", {}).get("sha256") != tok_sha:
        raise SystemExit(f"{manifest_path} does not describe {tok_path}; re-run prepare")
    data_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    torch.manual_seed(args.seed)
    device = get_device()
    os.makedirs(RUNS, exist_ok=True)

    base = Config(seq_len=args.seq_len, ple_dim=args.ple_dim, vocab_size=args.vocab,
                  d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads)
    model = make_model(args.arm, args.target_core, base, fixed_ffn=args.fixed_ffn).to(device)
    budget = model.param_budget()
    cfg = model.cfg

    # No weight decay on 1-D params (norms) or on lookup tables.
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "table" in n or "tok_emb" in n else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    train_b = Batcher("train", args.batch_size, args.seq_len, device, dataset,
                      seed=args.seed)
    val_b = Batcher("val", args.batch_size, args.seq_len, device, dataset)

    name = run_name(args)
    resume_path = os.path.join(RUNS, f"{name}.resume.pt")
    start_step, history, best, elapsed_before = 0, [], float("inf"), 0.0
    if args.resume:
        start_step, history, best, elapsed_before = resume_checkpoint(
            model, opt, cfg, args, name, tok_sha, data_manifest_sha, train_b, device)
        print(f"{name} RESUME step {start_step}/{args.steps} "
              f"after {elapsed_before:.0f}s", flush=True)
    t0 = time.time()

    def elapsed():
        return elapsed_before + time.time() - t0

    def save_resume(next_step):
        # Preserve the previous checkpoint until the next one is fully written;
        # an interrupt or full disk must not turn a resumable run into a corrupt
        # pickle.
        temporary = resume_path + ".partial"
        torch.save(checkpoint_payload(
            model, opt, cfg, args, name, tok_sha, data_manifest_sha, manifest,
            next_step, history, best, elapsed(), train_b, device), temporary)
        os.replace(temporary, resume_path)

    next_step = start_step
    try:
        for step in range(start_step, args.steps):
            lr = lr_at(step, args.steps, args.lr, args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = train_b()
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            next_step = step + 1

            if step % args.eval_every == 0 or step == args.steps - 1:
                vl = evaluate(model, val_b, args.eval_iters)
                best = min(best, vl)
                tok = next_step * args.batch_size * args.seq_len
                history.append({"step": step, "tokens": tok, "train": loss.item(), "val": vl})
                print(
                    f"{name} step {step:5d} | tok {tok / 1e6:6.1f}M | train {loss.item():.4f} "
                    f"| val {vl:.4f} | ppl {math.exp(vl):7.2f} | {elapsed():5.0f}s",
                    flush=True,
                )
            if next_step % args.save_every == 0 or next_step == args.steps:
                save_resume(next_step)
    except KeyboardInterrupt:
        save_resume(next_step)
        print(f"\n{name} INTERRUPTED at step {next_step}/{args.steps}; "
              f"resume with --resume {resume_path}", flush=True)
        return

    if not history:
        raise SystemExit("resume checkpoint has no validation history")

    result = {
        "arm": args.arm,
        "seed": args.seed,
        "tag": args.tag,
        "config": {k: v for k, v in cfg.__dict__.items()},
        "training": {
            "batch_size": args.batch_size,
            "steps": args.steps,
            "lr": args.lr,
            "warmup": args.warmup,
            "eval_every": args.eval_every,
            "eval_iters": args.eval_iters,
            "target_core": args.target_core,
            "fixed_ffn": args.fixed_ffn,
            "seed": args.seed,
            "save_every": args.save_every,
        },
        "tokenizer_sha256": tok_sha,
        "data_manifest_sha256": data_manifest_sha,
        "dataset": manifest.get("dataset"),
        "dataset_revision": manifest.get("revision"),
        "params": budget,
        "final_val": history[-1]["val"],
        "best_val": best,
        "final_ppl": math.exp(history[-1]["val"]),
        "tokens_seen": args.steps * args.batch_size * args.seq_len,
        "steps": args.steps,
        "wall_seconds": elapsed(),
        "history": history,
    }
    with open(os.path.join(RUNS, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    # Final checkpoints remain sample/export compatible, and additionally carry
    # resume state so a completed run can be inspected or re-opened safely.
    torch.save(checkpoint_payload(
        model, opt, cfg, args, name, tok_sha, data_manifest_sha, manifest,
        args.steps, history, best, elapsed(), train_b, device),
        os.path.join(RUNS, f"{name}.pt"))
    Path(resume_path).unlink(missing_ok=True)
    print(f"{name} DONE core={budget['core']:,} table={budget['table']:,} "
          f"val={result['final_val']:.4f} ppl={result['final_ppl']:.2f}")


if __name__ == "__main__":
    main()
