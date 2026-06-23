#!/usr/bin/env python3
"""Export a trained run (model.pt) into artifacts the Rust engine can read.

The Rust side never touches the pickle. Run this once per model; it writes into <out>/:
  model.safetensors  every weight tensor, f32 + contiguous (tied weights de-aliased by clone)
  meta.json          the run's GPTConfig kwargs + val_loss (the model shape)
  tokenizer.json     copied from the run dir, so encode/decode match training

Usage:
  python export/export.py <run_dir> [-o weights]
  e.g. python export/export.py ../sweeps/14_rollback/a149d94-variantbest-d512_valloss1.1479
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="directory containing model.pt + tokenizer.json")
    ap.add_argument("-o", "--out", default="weights", help="output directory (default: weights)")
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    # clone() de-aliases tied tensors (head.weight shares storage with token_emb.weight),
    # which safetensors otherwise rejects; .float().contiguous() gives Rust plain f32 row-major.
    tensors = {k: v.float().contiguous().clone() for k, v in sd.items()}
    save_file(tensors, str(out / "model.safetensors"))

    meta = dict(ckpt.get("config", {}))
    meta["val_loss"] = ckpt.get("val_loss")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    tok = run / "tokenizer.json"
    if tok.exists():
        shutil.copy(tok, out / "tokenizer.json")
    else:
        print(f"warning: no tokenizer.json in {run}")

    print(f"wrote {out}/model.safetensors ({len(tensors)} tensors), meta.json, tokenizer.json")


if __name__ == "__main__":
    main()
