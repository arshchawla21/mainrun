"""
Sweep runner.

Each entry in SWEEP is a dict of Hyperparameters fields to override for that run.
For every run we train, then save_run() drops a folder under  sweeps/  containing
the config, the .pt checkpoint, the tokenizer, and a copy of the log.

Run with:  task sweep
"""
import gc

import torch
from dataclasses import replace

from train import Hyperparameters, main, save_run, _git_short_hash

# --- define the sweep here: one dict per run, override any Hyperparameters field ---
SWEEP = [
    # {"vocab_size": 4_000},
    # {"vocab_size": 8_000},
    # {"vocab_size": 16_000},
    {"vocab_size": 32_000}
]

# flip on to log every run to W&B (grouped by project, one run name per config)
USE_WANDB = False
WANDB_PROJECT = "mainrun-sweep"


def tag_for(overrides: dict) -> str:
    """e.g. {'vocab_size': 8000} -> 'a1b2c3d-vocab_size8000'  (code version + what changed)."""
    params = "-".join(f"{k}{v}" for k, v in overrides.items()) or "run"
    return f"{_git_short_hash()}-{params}"


def run_sweep():
    base = Hyperparameters()
    for overrides in SWEEP:
        args = replace(base, **overrides)
        tag = tag_for(overrides)
        if USE_WANDB:
            args = replace(args, wandb=True, wandb_project=WANDB_PROJECT, wandb_run_name=tag)

        print(f"\n=== sweep run: {overrides} ===")
        result = main(args)
        save_run(result, base_dir="sweeps", tag=tag)

        # free GPU memory before the next run
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    run_sweep()