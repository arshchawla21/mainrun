"""
Sweep runner + plotting.

Each entry in SWEEP is a dict of Hyperparameters fields to override for that run.
For every run we train, save_run() drops a folder (incl. a copy of loss.log), then
we read that loss.log to plot:
  A) per run   -> loss_train_val.png  (train vs val, both nats/char)
  B) per sweep -> sweep_val_loss.png  (val loss of every run, one line each)

Run the sweep:   task sweep            (or: python3 sweep.py)
Just re-plot:     python3 sweep.py replot [sweeps/<campaign_dir>]
                  (rebuilds the pngs from existing loss.log files; no training)
"""
import gc
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")            # headless: render to file, no display
import matplotlib.pyplot as plt

from train import Hyperparameters, main, save_run, _git_short_hash

SWEEP_NAME = "vocab"   # campaign label
SWEEPS_DIR = Path(__file__).resolve().parent.parent / "sweeps"
SWEEP_DIR  = SWEEPS_DIR / f"{time.strftime('%Y%m%d')}_{SWEEP_NAME}"

# --- define the sweep here: one dict per run, override any Hyperparameters field ---
SWEEP = [
    {"vocab_size": 4_000},
    {"vocab_size": 8_000},
    {"vocab_size": 16_000},
    {"vocab_size": 24_000},
    {"vocab_size": 32_000},
]

# flip on to log every run to W&B (grouped by project, one run name per config)
USE_WANDB = False
WANDB_PROJECT = "mainrun-sweep"


def tag_for(overrides: dict) -> str:
    """e.g. {'vocab_size': 8000} -> 'a1b2c3d-vocab_size8000'  (code version + what changed)."""
    params = "-".join(f"{k}{v}" for k, v in overrides.items()) or "run"
    return f"{_git_short_hash()}-{params}"


def label_for(overrides: dict) -> str:
    """Short legend label, e.g. 'vocab_size=8000'."""
    return ", ".join(f"{k}={v}" for k, v in overrides.items()) or "run"


def read_loss_log(path):
    """Parse a loss.log (JSON lines: step, epoch, train, val) into a list of dicts."""
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


# ---- A) per-run: train vs val, both nats/char (read from loss.log) ----
def plot_run(loss_log, run_dir, title=""):
    recs = read_loss_log(loss_log)
    if not recs:
        print(f"no loss data in {loss_log}")
        return
    steps = [r["step"] for r in recs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, [r["train"] for r in recs], "-", color="#378ADD", label="train")
    ax.plot(steps, [r["val"]   for r in recs], "-", color="#D85A30", label="val")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (nats/char)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = Path(run_dir) / "loss_train_val.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


# ---- B) per-sweep: val loss of every run (x = epoch, so runs line up) ----
def plot_sweep_val(runs, out_dir, title=""):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, loss_log in runs:
        recs = read_loss_log(loss_log)
        if not recs:
            continue
        ax.plot([r["epoch"] for r in recs], [r["val"] for r in recs], "-", label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val loss (nats/char)")
    ax.set_title(f"{title} — validation loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(out_dir) / "sweep_val_loss.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def run_sweep():
    base = Hyperparameters()
    sweep_runs = []   # (label, loss_log_path) -> feeds plot B
    for overrides in SWEEP:
        args = replace(base, **overrides)
        tag  = tag_for(overrides)
        if USE_WANDB:
            args = replace(args, wandb=True, wandb_project=WANDB_PROJECT, wandb_run_name=tag)

        print(f"\n=== sweep run: {overrides} ===")
        result = main(args)
        run_dir = save_run(result, base_dir=str(SWEEP_DIR), tag=tag)

        loss_log = Path(run_dir) / "loss.log"
        plot_run(loss_log, run_dir, title=label_for(overrides))   # A
        sweep_runs.append((label_for(overrides), loss_log))

        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plot_sweep_val(sweep_runs, SWEEP_DIR, title=SWEEP_NAME)        # B
    print(f"\nsweep done -> {SWEEP_DIR}")


# ---------- re-plot an existing campaign from disk (no training) ----------
def _run_dirs(sweep_dir):
    sweep_dir = Path(sweep_dir)
    return [d for d in sweep_dir.iterdir() if d.is_dir() and (d / "loss.log").exists()]


def _labels_and_swept(run_dirs):
    """Label each run from config.json, keyed on whatever hyperparams differ across runs."""
    configs = {}
    for d in run_dirs:
        cfg = d / "config.json"
        configs[d] = json.loads(cfg.read_text()) if cfg.exists() else {}
    common = set.intersection(*[set(c) for c in configs.values()]) if configs else set()
    swept = [k for k in sorted(common)
             if len({json.dumps(configs[d][k]) for d in run_dirs}) > 1]
    labels = {d: (", ".join(f"{k}={configs[d][k]}" for k in swept) if swept else d.name)
              for d in run_dirs}
    return labels, swept, configs


def replot(sweep_dir):
    """Rebuild plot A for every run + plot B for the campaign, from existing loss.log files."""
    sweep_dir = Path(sweep_dir)
    run_dirs = _run_dirs(sweep_dir)
    if not run_dirs:
        print(f"no runs with loss.log under {sweep_dir}")
        return
    labels, swept, configs = _labels_and_swept(run_dirs)
    run_dirs.sort(key=lambda d: tuple(configs[d].get(k) for k in swept) if swept else d.name)

    runs = []
    for d in run_dirs:
        loss_log = d / "loss.log"
        plot_run(loss_log, d, title=labels[d])          # A (overwrites png)
        runs.append((labels[d], loss_log))
    plot_sweep_val(runs, sweep_dir, title=sweep_dir.name)   # B
    print(f"replotted {len(runs)} runs -> {sweep_dir}")


def _latest_campaign():
    dirs = [d for d in SWEEPS_DIR.iterdir() if d.is_dir()] if SWEEPS_DIR.exists() else []
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "replot":
        target = Path(sys.argv[2]) if len(sys.argv) > 2 else _latest_campaign()
        if target is None:
            print("no campaign found to replot")
        else:
            replot(target)
    else:
        run_sweep()