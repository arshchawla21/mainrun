"""
Load a finished run (model + tokenizer) from a run directory produced by train.save_run().

A run dir contains:
  - model.pt        : {model_state_dict, config (GPTConfig kwargs), args, val_loss, step}
  - tokenizer.json  : the exact tokenizer used at train time (load it; never retrain)
  - config.json     : the Hyperparameters used (human-readable; not needed to rebuild)

Loading tokenizer.json directly is important: it preserves the precise pre-tokenizer/merges,
including superbpe's whitespace-free ByteLevel, so encode/decode match training exactly without
needing the fork, the corpus, or the two-stage training procedure at sample time.
"""
from pathlib import Path

import torch
from tokenizers import Tokenizer as HFTokenizer

from model.gpt import GPTConfig, GPT
from model.tokenizer import Tokenizer


def load_run(run_dir, device="cpu"):
    """Return (model, tokenizer, ckpt) ready for inference. model is .eval() on `device`;
    tokenizer is the Tokenizer wrapper; ckpt is the raw checkpoint dict (val_loss, args, ...)."""
    run_dir = Path(run_dir)

    model_path = run_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"no model.pt in {run_dir}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    tok_path = run_dir / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"no tokenizer.json in {run_dir} (saved alongside model.pt by save_run)")
    tok = Tokenizer(HFTokenizer.from_file(str(tok_path)))

    return model, tok, ckpt
