"""
Starter visualiser -- bench copies this into a new campaign folder as visualise.py.
EDIT ME freely: I read the folder I sit in and plot it however you like.

Run:  python3 visualise.py     (from inside the campaign folder)
"""
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "mainrun"))   # so `import plots` works from here
from plots import Campaign, Plot

c = Campaign(HERE)

# primary x-axis + optional overlay group: from results.json meta if present, else fall back to N
x, group = "N", None
res = HERE / "results.json"
if res.exists():
    meta = json.loads(res.read_text())
    if isinstance(meta, dict):
        x = meta.get("x") or x
        group = meta.get("group")
rows = c.sorted(x if (c.runs and x in c.runs[0]) else "N")
x = x if (rows and x in rows[0]) else "N"

# --- final loss vs the swept axis (evenly spaced so relative gaps read clearly) ---
xcat = sorted({r[x] for r in rows})
p = Plot(title=f"{c.name}: val vs {x}", xlabel=x, ylabel="val loss (nats/char)", xcat=xcat)
if group:                                   # one line per group value (e.g. token_type)
    for gval in sorted({r[group] for r in rows}):
        g = [r for r in rows if r[group] == gval]
        p.line([r[x] for r in g], [r["val"] for r in g], label=f"{group}={gval}")
else:
    p.line([r[x] for r in rows], [r["val"] for r in rows])
    p.annotate([r[x] for r in rows], [r["val"] for r in rows], [f"{r['val']:.4f}" for r in rows])
p.save(HERE / "val_vs_x.png")

# --- training curves ---
pe = Plot(title=f"{c.name}: val vs epoch", xlabel="epoch", ylabel="val loss (nats/char)")
for r in rows:
    ep, _, vl = c.loss_curve(r)
    if ep:
        lbl = f"{group}={r.get(group)}, {x}={r.get(x)}" if group else f"{x}={r.get(x)}"
        pe.line(ep, vl, label=lbl, fmt="-")
pe.save(HERE / "val_vs_epoch.png")

best = c.best()
if best:
    extra = f"{group}={best.get(group)}, " if group else ""
    print(f"best: {extra}{x}={best.get(x)}  val={best['val']:.4f}")
