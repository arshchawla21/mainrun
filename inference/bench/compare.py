#!/usr/bin/env python3
"""Compare CPU inference speed: the Rust engine vs the reference PyTorch model.

Both decode the same number of greedy tokens from the same prompt, on CPU, and we report
tokens/sec for each (load time excluded). Note the comparison is not purely language vs
language: the Rust engine uses a KV cache (incremental decode), while PyTorch's
`model.generate` recomputes the whole context each step (no cache, O(T^2)). So the speedup
reflects both the language and the algorithm -- which is exactly the point of a
purpose-built engine. Both numbers are labelled honestly.

Usage:
  python bench/compare.py <run_dir> [--weights weights] [--tokens 128] [--warmup 16]
  e.g. python bench/compare.py ../sweeps/14_rollback/a149d94-variantbest-d512_valloss1.1479
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # inference/bench
INFER = HERE.parent                             # inference
MAINRUN = INFER.parent / "mainrun"              # the training code (model.load lives here)
sys.path.insert(0, str(MAINRUN))

import torch  # noqa: E402
from model.load import load_run  # noqa: E402


def bench_torch(run_dir, prompt, tokens, warmup):
    """Time PyTorch CPU generation (decode loop only, load excluded)."""
    torch.set_grad_enabled(False)
    model, tok, _ = load_run(run_dir, device="cpu")
    eos = tok.stoi.get("<eos>", None)
    ids = tok.encode(prompt) or [eos]
    idx = torch.tensor([ids], dtype=torch.long)

    # warmup (lets threads spin up, caches warm), not timed
    if warmup:
        model.generate(idx, max_new_tokens=warmup, temperature=1.0, top_k=None)

    start = time.perf_counter()
    model.generate(idx, max_new_tokens=tokens, temperature=1.0, top_k=None)
    total_s = time.perf_counter() - start
    return {
        "engine": "pytorch",
        "tokens": tokens,
        "total_s": total_s,
        "tok_per_s": tokens / total_s,
        "ms_per_tok": total_s * 1000.0 / tokens,
        "threads": torch.get_num_threads(),
    }


def bench_rust(binary, weights, prompt, tokens, warmup):
    """Run the Rust `bench` subcommand and parse its JSON line from stdout."""
    out = subprocess.run(
        [str(binary), "--weights", str(weights), "bench",
         "--prompt", prompt, "--tokens", str(tokens), "--warmup", str(warmup)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="sweep run dir (for the PyTorch model + tokenizer)")
    ap.add_argument("--weights", default=str(INFER / "weights"), help="Rust export dir")
    ap.add_argument("--binary", default=str(INFER / "target/release/inference"))
    ap.add_argument("--prompt", default="Show HN: ")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=16)
    args = ap.parse_args()

    if not Path(args.binary).exists():
        sys.exit(f"rust binary not found at {args.binary} -- run `cargo build --release` first")

    print(f"prompt={args.prompt!r}  tokens={args.tokens}  warmup={args.warmup}\n")

    rust = bench_rust(args.binary, args.weights, args.prompt, args.tokens, args.warmup)
    torch_res = bench_torch(args.run_dir, args.prompt, args.tokens, args.warmup)

    print(f"{'engine':<10}{'tok/s':>10}{'ms/tok':>10}{'total s':>10}")
    print("-" * 40)
    for r in (rust, torch_res):
        print(f"{r['engine']:<10}{r['tok_per_s']:>10.1f}{r['ms_per_tok']:>10.2f}{r['total_s']:>10.3f}")
    print("-" * 40)
    speedup = rust["tok_per_s"] / torch_res["tok_per_s"]
    print(f"\nrust is {speedup:.1f}x faster (KV-cache decode vs PyTorch recompute, "
          f"{torch_res['threads']} torch threads)")


if __name__ == "__main__":
    main()
