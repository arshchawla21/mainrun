"""
Plotting toolkit for sweep campaigns -- the reusable "house style", nothing campaign-specific.

stdlib + matplotlib (+ numpy/scipy for the fit) ONLY -- deliberately no torch/training import,
so it loads instantly and can be called from any campaign folder.

Two pieces:
  Campaign(dir) : loads a campaign folder by scanning its run sub-dirs (config.json + loss.log +
                  mainrun.log). Source of truth is the folders, so it works on partial/ongoing
                  campaigns that have no results.json yet.
  Plot(...)     : a styled matplotlib figure (our grid/size/palette) with fluent
                  line / scatter / annotate / fit_power / save helpers.

Custom plotting lives in each  sweeps/<campaign>/visualise.py , which reads the folder it sits in
and calls these. See _visualise_template.py for the starter bench drops into new campaigns.
"""
import re
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================ param accounting (torch-free) ============================
def matmul_params(cfg):
    """Transformer + tied-embedding matmul params (the count isoparam/find_n call 'N')."""
    d, L, V = cfg["d_model"], cfg["n_layer"], cfg["vocab_size"]
    return 12 * L * d * d + V * d


def param_count(cfg):
    """Approx true param count: matmuls + (positional + per-layer bias/LN + final LN).
    Matches the trainer's sum(p.numel()) to ~0.1% for this architecture."""
    d, L, V = cfg["d_model"], cfg["n_layer"], cfg["vocab_size"]
    T = cfg.get("block_size", 0)
    return 12 * L * d * d + (V + T + 13 * L + 2) * d


def emb_fraction(cfg):
    """Share of the matmul budget spent on the (lookup) embedding/output table."""
    d, V = cfg["d_model"], cfg["vocab_size"]
    return round(V * d / matmul_params(cfg), 3)


def flops_from_steps(cfg, steps):
    """Approx fwd+bwd training FLOPs from the step count (torch-free; no tokenizer needed).
    D = steps * batch * block tokens processed."""
    if not steps:
        return None
    d, L = cfg["d_model"], cfg["n_layer"]
    D = steps * cfg["batch_size"] * cfg["block_size"]
    return 6 * matmul_params(cfg) * D + 12 * L * cfg["block_size"] * d * D


def fmt_si(v):
    """Axis-label formatter: 32_000_000 -> '32M', 64000 -> '64k'."""
    if isinstance(v, (int, float)) and abs(v) >= 1e6:
        return f"{v/1e6:g}M"
    if isinstance(v, (int, float)) and abs(v) >= 1e3 and v % 1e3 == 0:
        return f"{v/1e3:g}k"
    return str(v)


# ============================ scaling-law fit ============================
def fit_scaling(Ns, vals):
    """Fit loss = a*N**-b + c with bounded, multi-restart curve_fit. Returns ((a,b,c), rmse)
    or None (e.g. scipy missing). Bounds keep the asymptote c from running away."""
    try:
        from scipy.optimize import curve_fit
    except Exception:
        print("  (scipy not installed -> skipping fit; pip install scipy)")
        return None
    Ns = np.asarray(Ns, float); vals = np.asarray(vals, float)
    f = lambda N, a, b, c: a * N ** (-b) + c
    lo_c = max(0.0, vals.min() - 0.6)
    bounds = ([0.0, 0.05, lo_c], [50.0, 0.7, vals.min()])
    best = None
    for b0 in (0.1, 0.2, 0.35, 0.5):
        a0 = max(1e-3, vals.max() - vals.min()) * Ns.min() ** b0
        try:
            popt, _ = curve_fit(f, Ns, vals, p0=[a0, b0, max(lo_c, vals.min() - 0.05)],
                                maxfev=200000, bounds=bounds)
        except Exception:
            continue
        sse = float(np.sum((f(Ns, *popt) - vals) ** 2))
        if best is None or sse < best[1]:
            best = (tuple(popt), sse)
    if best is None:
        return None
    popt, sse = best
    return popt, (sse / len(Ns)) ** 0.5


def knee(popt, n_lo, n_hi, tol=0.01):
    """Smallest N where doubling N buys < tol nats/char. None if it never flattens."""
    a, b, c = popt
    f = lambda N: a * N ** (-b) + c
    N = float(n_lo)
    while N < n_hi * 4:
        if f(N) - f(2 * N) < tol:
            return N
        N *= 1.15
    return None


# ============================ campaign loader ============================
class Campaign:
    """Loads a campaign folder from its run sub-dirs. Each run dict carries the run's config
    plus computed N / emb_frac / val / dir / its loss.log path."""

    def __init__(self, path):
        self.dir = Path(path).resolve()
        self.name = self.dir.name
        self.runs = []
        for run in sorted(self.dir.iterdir()):
            cfg_p = run / "config.json"
            if not run.is_dir() or not cfg_p.exists():
                continue
            cfg = json.loads(cfg_p.read_text())
            t, steps = self._train_meta(run)
            self.runs.append({**cfg, "dir": run, "losslog": run / "loss.log",
                              "N": param_count(cfg), "emb_frac": emb_fraction(cfg),
                              "val": self._final_val(run),
                              "time": t, "flops": flops_from_steps(cfg, steps)})

    @staticmethod
    def _train_meta(run):
        """(elapsed_time_s, total_steps) from the last step logged in mainrun.log."""
        main = run / "mainrun.log"
        if not main.exists():
            return None, None
        last = None
        for l in main.read_text().splitlines():
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("event") in ("training_step", "validation_step"):
                last = r
        if not last:
            return None, None
        return last.get("elapsed_time"), last.get("max_steps") or last.get("step")

    @staticmethod
    def _final_val(run):
        log = run / "loss.log"                      # 1) last val in loss.log
        if log.exists():
            recs = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
            if recs and "val" in recs[-1]:
                return recs[-1]["val"]
        main = run / "mainrun.log"                  # 2) last validation_step in mainrun.log
        if main.exists():
            last = None
            for l in main.read_text().splitlines():
                try:
                    r = json.loads(l)
                except Exception:
                    continue
                if r.get("event") == "validation_step":
                    last = r
            if last:
                return last.get("loss")
        m = re.search(r"valloss([0-9.]+)", run.name)  # 3) the folder name
        return float(m.group(1)) if m else None

    def sorted(self, key, reverse=False):
        return sorted([r for r in self.runs if r.get(key) is not None],
                      key=lambda r: r[key], reverse=reverse)

    def best(self, key="val"):
        rows = [r for r in self.runs if r.get(key) is not None]
        return min(rows, key=lambda r: r[key]) if rows else None

    def loss_curve(self, run):
        """(epochs, train, val) lists for one run; empty lists if it has no loss.log."""
        log = run["losslog"]
        if not Path(log).exists():
            return [], [], []
        recs = [json.loads(l) for l in Path(log).read_text().splitlines() if l.strip()]
        return ([r["epoch"] for r in recs],
                [r.get("train") for r in recs],
                [r.get("val") for r in recs])


# ============================ styled figure ============================
class Plot:
    """A matplotlib figure in our house style. Fluent: every method returns self (except fit).

    Pass `xcat=[ordered category values]` for an EVENLY-SPACED categorical x-axis (e.g. model
    sizes 32M..192M get equal gaps instead of log-bunched) -- then line/scatter/annotate take the
    raw category values for xs and they're placed at 0,1,2,... with formatted tick labels."""
    PALETTE = ["#378ADD", "#D85A30", "#4CAF50", "#9C27B0", "#FF9800", "#00838F", "#795548"]

    def __init__(self, title="", xlabel="", ylabel="", logx=False, logy=False,
                 xcat=None, figsize=(8, 5), ax=None):
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.fig, self.ax = ax.figure, ax
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        if logx:
            self.ax.set_xscale("log")
        if logy:
            self.ax.set_yscale("log")
        self._cat = None
        if xcat is not None:                       # evenly-spaced categorical x
            self._cat = {v: i for i, v in enumerate(xcat)}
            self.ax.set_xticks(range(len(xcat)))
            self.ax.set_xticklabels([fmt_si(v) for v in xcat])
            self.ax.set_xlim(-0.4, len(xcat) - 0.6)
        self.ax.grid(True, alpha=0.3, which="both")
        self._ci = 0

    def _X(self, xs):
        return [self._cat[x] for x in xs] if self._cat is not None else list(xs)

    def _color(self, c=None):
        if c:
            return c
        c = self.PALETTE[self._ci % len(self.PALETTE)]
        self._ci += 1
        return c

    def line(self, xs, ys, label=None, color=None, fmt="o-"):
        self.ax.plot(self._X(xs), list(ys), fmt, color=self._color(color), label=label, zorder=3)
        return self

    def scatter(self, xs, ys, label=None, color=None, s=30):
        self.ax.scatter(self._X(xs), list(ys), color=self._color(color), s=s, label=label, zorder=3)
        return self

    def annotate(self, xs, ys, texts, dy=9, fontsize=8):
        for x, y, t in zip(self._X(xs), ys, texts):
            self.ax.annotate(str(t), (x, y), textcoords="offset points",
                             xytext=(0, dy), fontsize=fontsize, ha="center")
        return self

    def xticks(self, ticks, labels):
        self.ax.set_xticks(list(ticks))
        self.ax.set_xticklabels(list(labels))
        return self

    def fit_power(self, Ns, vals, color="#666", label=None, show_knee=True):
        """Overlay loss = a*N**-b + c (+ knee line). Returns ((a,b,c), rmse) or None.
        No-op on a categorical x-axis (a continuous curve has no place there)."""
        if self._cat is not None:
            return None
        res = fit_scaling(Ns, vals)
        if not res:
            return None
        (a, b, c), rmse = res
        grid = np.logspace(np.log10(min(Ns)), np.log10(max(Ns) * 1.5), 100)
        lbl = label if label is not None else f"{a:.2g}·N^-{b:.3f}+{c:.3f}  (rmse {rmse:.4f})"
        self.ax.plot(grid, a * grid ** (-b) + c, "--", color=color, label=lbl)
        if show_knee:
            k = knee((a, b, c), min(Ns), max(Ns))
            if k:
                self.ax.axvline(k, color=color, ls=":", alpha=0.5)
        return (a, b, c), rmse

    def save(self, path):
        if self.ax.get_legend_handles_labels()[1]:
            self.ax.legend(fontsize=8)
        self.fig.tight_layout()
        self.fig.savefig(path, dpi=150)
        plt.close(self.fig)
        print(f"saved {path}")
