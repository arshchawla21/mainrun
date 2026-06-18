"""
Unified benchmark engine. One generic run loop drives every sweep; the only thing that
changes between experiments is the *config* in experiments.py.

A Sweep declares:
  axes     : the variable(s) to sweep (cartesian product -> one run per point)
  hold     : Hyperparameters held fixed for the whole campaign
  resolve  : how to derive dependent params (d_model/n_head) from each point
  x/group  : how rows map onto plot axes
  plots    : which plot views to render (see plots.py registry)

CLI:
  python3 bench.py run  <experiment>            # train the sweep (resumes: skips cells already in results.json)
  python3 bench.py run  <experiment> --force    # retrain every cell, ignoring cached results
  python3 bench.py plot <experiment> [dir]      # re-plot an existing campaign (no training)
  python3 bench.py list                          # show available experiments

Design notes:
  * solve_width() is the single d_model/n_head solver that all four legacy scripts needed.
    `budget` is either a fixed param count (iso-param sweeps) or the *name* of a swept axis
    such as "target_N" (size / find-N sweeps). n_head is auto-picked for head_dim ~= 64,
    which is free (changes neither N nor the dominant 6*N*D FLOPs) and keeps shape default-ish.
  * A one-off comparison (e.g. MHA vs MLA) is just a sweep with one short axis -- no new script.
"""
import gc
import json
import itertools
import dataclasses
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

import torch

from train import (Hyperparameters, main, save_run, _git_short_hash,
                   get_titles, train_tokenizer, Tokenizer)

SWEEPS_DIR = Path(__file__).resolve().parent.parent / "sweeps"
_FIELDS = {f.name for f in dataclasses.fields(Hyperparameters)}


# ============================ the config schema ============================
@dataclass
class Sweep:
    name: str                                   # campaign name -> sweeps/<date>_<name>/
    axes: dict                                  # {key: [values]}; cartesian product = the runs
    resolve: Optional[Callable] = None          # (merged_point_dict) -> derived field overrides
    hold: dict = field(default_factory=dict)    # fields fixed across the whole sweep
    x: Optional[str] = None                     # primary swept axis -> stored in results.json
    group: Optional[str] = None                 # overlay grouping key -> stored in results.json
    tokens_per_step: int = 4096                 # held constant -> no batch/step confound


# ============================ derived-param resolvers ============================
def shape_for_budget(target, vocab, n_layer, head_dim=64):
    """Solve 12*n_layer*d^2 + vocab*d = target for d_model, then pick n_head so
    d_model/n_head ~= head_dim (n_head even, d_model divisible by n_head)."""
    a, b, c = 12 * n_layer, vocab, -target
    d_raw = (-b + (b * b - 4 * a * c) ** 0.5) / (2 * a)
    n_head = max(4, round(d_raw / head_dim))
    if n_head % 2:
        n_head += 1
    d_model = max(n_head, round(d_raw / n_head) * n_head)
    return d_model, n_head


def est_params(vocab, d_model, n_layer):
    """(transformer matmul params, embedding/tied-output params, total)."""
    xf = 12 * n_layer * d_model ** 2
    emb = vocab * d_model
    return xf, emb, xf + emb


def solve_width(budget, head_dim=64):
    """Resolver: hit a parameter `budget` by solving d_model (+ n_head) at the point's
    vocab/n_layer. `budget` is a number (fixed N) or the name of an axis key (e.g. 'target_N')."""
    def resolve(p):
        target = p[budget] if isinstance(budget, str) else budget
        d, nh = shape_for_budget(target, p["vocab_size"], p["n_layer"], head_dim)
        return {"d_model": d, "n_head": nh}
    return resolve


def heads_for(head_dim=64):
    """Resolver for free architecture sweeps: d_model + n_layer come straight from the axes
    (params are NOT pinned to a budget -> N is an output, plotted on x), this only derives
    n_head so d_model/n_head ~= head_dim. Pick d_model divisible by head_dim (e.g. multiples of 64)."""
    def resolve(p):
        d = p["d_model"]
        nh = max(1, round(d / head_dim))
        while d % nh:                      # guarantee d_model % n_head == 0 (GPT asserts this)
            nh -= 1
        return {"n_head": nh}
    return resolve


# ============================ FLOP / token accounting ============================
def train_flops(vocab, d_model, n_layer, block_size, tokens):
    """Approx fwd+bwd training FLOPs; constants are consistent across configs so relative
    (iso-FLOP) comparisons hold. Output projection counted as a matmul even when tied."""
    matmul_params = 12 * n_layer * d_model ** 2 + vocab * d_model
    flops = 6 * matmul_params * tokens
    flops += 12 * n_layer * block_size * d_model * tokens
    return flops


_titles_cache = {}
_tokens_cache = {}
def measure_train_tokens(vocab, base, token_type=None, transition_ratio=None):
    """Exact train-token count for a (vocab, token_type) (tokenises like main()). Cached."""
    token_type = token_type or base.token_type
    transition_ratio = base.transition_ratio if transition_ratio is None else transition_ratio
    tkey = (base.num_titles, base.seed, base.val_frac)
    if tkey not in _titles_cache:
        _titles_cache[tkey] = get_titles(base.num_titles, base.seed, base.val_frac)
    # transition_ratio only affects superbpe, but include it in the key so token counts stay correct.
    ckey = (vocab, token_type, transition_ratio, *tkey)
    if ckey not in _tokens_cache:
        train_titles, val_titles = _titles_cache[tkey]
        eos = "<eos>"
        tok = Tokenizer(train_tokenizer(token_type, train_titles + val_titles, vocab, eos_token=eos,
                                        transition_ratio=transition_ratio))
        _tokens_cache[ckey] = len(tok.encode(eos.join(train_titles) + eos))
    return _tokens_cache[ckey]


def read_loss_log(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


# ============================ plan building ============================
def _points(axes):
    keys = list(axes)
    for combo in itertools.product(*(axes[k] for k in keys)):
        yield dict(zip(keys, combo))


def _overrides(sweep, base, point):
    """Concrete Hyperparameters overrides for one point: held fields + real-field axis
    values + resolver output + an auto batch_size (so tokens/step stays constant, unless a
    sweep sets batch_size explicitly -- e.g. the batch sweep, which varies tokens/step on purpose)."""
    merged = {**vars(base), **sweep.hold, **point}
    ov = {k: v for k, v in sweep.hold.items() if k in _FIELDS}
    ov.update({k: v for k, v in point.items() if k in _FIELDS})
    if sweep.resolve:
        ov.update(sweep.resolve(merged))
    # shape may be unset (e.g. a baseline with no resolver) -> fall back to the defaults,
    # so downstream code (plan table, tagging, rows) can always read d_model/n_head.
    ov.setdefault("d_model", merged["d_model"])
    ov.setdefault("n_head", merged["n_head"])
    # auto batch_size to pin tokens/step -- but only if the sweep didn't set batch_size itself.
    bs = ov.get("block_size", merged["block_size"])
    ov.setdefault("batch_size", max(1, sweep.tokens_per_step // bs))
    return ov


def _metrics(base, point, ov):
    """Derived columns every row carries: FLOPs, embedding fraction, head_dim."""
    p = {**vars(base), **point, **ov}
    v, d, nl, bs = p["vocab_size"], p["d_model"], p["n_layer"], p["block_size"]
    toks = p["epochs"] * measure_train_tokens(v, base, p["token_type"], p["transition_ratio"])
    _, emb, tot = est_params(v, d, nl)
    return {"flops": train_flops(v, d, nl, bs, toks),
            "emb_frac": round(emb / tot, 3),
            "head_dim": d // p["n_head"],
            "tokens": toks}


def _fmt(v):
    if isinstance(v, (int, float)) and v >= 1e6:
        return f"{v/1e6:g}M"
    if isinstance(v, (int, float)) and v >= 1e3 and v % 1e3 == 0:
        return f"{v/1e3:g}k"
    return str(v)


def _label(point):
    return ", ".join(f"{k}={_fmt(v)}" for k, v in point.items())


def _tag(point, ov):
    parts = "-".join(f"{k}{_fmt(v)}" for k, v in point.items())
    return f"{_git_short_hash()}-{parts}-d{ov['d_model']}"


def print_plan(sweep, base, plan):
    print(f"[{sweep.name}]  {len(plan)} runs | hold={sweep.hold or '{}'} "
          f"| tokens/step={sweep.tokens_per_step}\n")
    pkeys = list(sweep.axes)
    head = "".join(f"{k:>12}" for k in pkeys)
    print(head + f"{'d_model':>8}{'n_head':>7}{'estN(M)':>9}{'emb%':>6}{'FLOPs':>12}")
    for point in plan:
        ov = _overrides(sweep, base, point)
        m = _metrics(base, point, ov)
        p = {**vars(base), **point, **ov}
        _, _, N = est_params(p["vocab_size"], p["d_model"], p["n_layer"])
        cells = "".join(f"{_fmt(point[k]):>12}" for k in pkeys)
        print(cells + f"{ov['d_model']:>8}{ov['n_head']:>7}{N/1e6:>9.1f}"
              f"{100*m['emb_frac']:>5.0f}%{m['flops']:>12.3e}")


# ============================ run / replot ============================
def _cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def campaign_dir(sweep):
    # one stable folder per experiment (no date) -> a campaign never repeats; resume always finds
    # it across days. The creation/run date is recorded in mainrun.log, not the folder name.
    return SWEEPS_DIR / sweep.name


def _point_key(sweep, d):
    """Identity of a sweep point = its axis values. Works on both a `point` dict and a saved
    row (rows carry the axis keys), so reruns can detect already-computed cells."""
    return tuple((k, d[k]) for k in sweep.axes)


def load_results(out):
    """Existing rows for a campaign (for resume/merge), or [] if none/unreadable."""
    p = out / "results.json"
    if p.exists():
        try:
            return json.loads(p.read_text()).get("rows", [])
        except (json.JSONDecodeError, OSError):
            return []
    return []


def run(sweep, force=False):
    base = Hyperparameters()
    out = campaign_dir(sweep)
    out.mkdir(parents=True, exist_ok=True)
    plan = list(_points(sweep.axes))
    print_plan(sweep, base, plan)

    # resume: keep prior rows and skip points already computed (unless --force retrains all).
    rows = [] if force else load_results(out)
    done = {_point_key(sweep, r) for r in rows}
    if done:
        print(f"\nresuming: {len(done)} cell(s) already done -> skipping them "
              f"({len(plan) - len(done)} to run){' [--force overrides]' if force else ''}")

    for point in plan:
        if _point_key(sweep, point) in done:
            print(f"  ~~ skip (cached): {_label(point)}")
            continue
        ov = _overrides(sweep, base, point)
        args = replace(base, **ov)
        print(f"\n=== {_label(point)}  (d_model={ov['d_model']}, n_head={ov['n_head']}, "
              f"batch={ov['batch_size']}) ===")
        try:
            result = main(args)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  !! OOM, skip {_label(point)}: {e}")
            _cuda_cleanup()
            continue
        N = sum(p.numel() for p in result["model"].parameters())
        run_dir = save_run(result, base_dir=str(out), tag=_tag(point, ov))
        rows.append({**point, **ov, **_metrics(base, point, ov),
                     "N": N, "val": result["val_loss"], "run_dir": str(run_dir)})
        print(f"  -> N={N/1e6:.1f}M  val={result['val_loss']:.4f}")
        save_results(sweep, rows, out)   # checkpoint after each cell -> a crash keeps progress
        del result
        _cuda_cleanup()

    save_results(sweep, rows, out)
    visualise(out)
    print(f"\ndone -> {out}")
    return rows


def save_results(sweep, rows, out):
    (out / "results.json").write_text(json.dumps(
        {"sweep": sweep.name, "x": sweep.x, "group": sweep.group, "rows": rows}, indent=2))


_TEMPLATE = Path(__file__).resolve().parent / "_visualise_template.py"
def visualise(out):
    """Plotting is per-campaign: scaffold sweeps/<dir>/visualise.py from the template (once),
    then run it. Edit that file for campaign-specific plots; it reads the folder it sits in."""
    import shutil
    import subprocess
    out = Path(out)
    if not out.is_dir():
        print(f"campaign dir not found: {out}")
        return
    vis = out / "visualise.py"
    if not vis.exists() and _TEMPLATE.exists():
        shutil.copy(_TEMPLATE, vis)
        print(f"scaffolded {vis} (edit it for custom plots)")
    if vis.exists():
        subprocess.run(["python3", "visualise.py"], cwd=str(out))


def latest_campaign(sweep):
    # campaigns are now date-free single folders, so there's exactly one per experiment.
    out = campaign_dir(sweep)
    return out if out.is_dir() else None


# ============================ CLI ============================
if __name__ == "__main__":
    import sys
    from experiments import EXPERIMENTS

    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        print("experiments:")
        for k, s in EXPERIMENTS.items():
            print(f"  {k:<12} -> {s.name}  (sweeps {', '.join(s.axes)})")
    elif cmd in ("run", "plot"):
        name = sys.argv[2]
        if name not in EXPERIMENTS:
            sys.exit(f"unknown experiment '{name}'. options: {', '.join(EXPERIMENTS)}")
        sweep = EXPERIMENTS[name]
        if cmd == "run":
            run(sweep, force="--force" in sys.argv[3:])
        else:
            target = Path(sys.argv[3]) if len(sys.argv) > 3 else latest_campaign(sweep)
            if target is None:
                sys.exit(f"no campaign found for '{name}' under {SWEEPS_DIR}")
            visualise(target)            # re-run that campaign's visualise.py (no training)
    else:
        sys.exit("usage: bench.py [list | run <exp> [--force] | plot <exp> [dir]]")
