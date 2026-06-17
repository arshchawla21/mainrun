"""
Experiment definitions -- THE file you edit to run benchmarks.

Each entry is a Sweep (see bench.py for the schema). To add a benchmark, add an entry;
to run it:  `python3 bench.py run <key>`  /  re-plot:  `python3 bench.py plot <key>`.

Conventions used below:
  * solve_width(budget=...) derives d_model + n_head for each point. `budget` is a fixed
    param count (iso-param sweeps) or the name of a swept axis like "target_N" (size/find-N).
  * axes with >1 key form a cartesian grid (e.g. find_n = vocab x target_N).
  * a one-off comparison (MHA vs MLA, dropout on/off, ...) is just a short axis -- no new file.
"""
from bench import Sweep, solve_width

EXPERIMENTS = {

    # --- baseline:
    "baseline": Sweep(
        name="baseline",
        axes={"vocab_size": [16_000]},
    ),

    # --- tokenizer benchmark: fix the ~96M param budget, identify strongest BPE vocab size.
    # 16/06/2026: found 64k to be strongest
    "isoparam": Sweep(
        name="vocab_isoparam_96M",
        axes={"vocab_size": [4_000, 8_000, 16_000, 32_000, 64_000, 96_000]},
        hold={"n_layer": 8},
        resolve=solve_width(budget=96_000_000, head_dim=64),
        x="vocab_size",
        tokens_per_step=4096,
    ),

    # --- size sweep at the fixed 64k vocab: hold depth, grow width to hit each N, find N*.
    # 17/06/2026
    "size": Sweep(
        name="size",
        axes={"target_N": [32_000_000, 64_000_000, 96_000_000, 128_000_000, 192_000_000]},
        hold={"vocab_size": 64_000, "n_layer": 8},
        resolve=solve_width(budget="target_N", head_dim=64),
        x="N",
        tokens_per_step=4096,
    ),

    # --- shape sweep: hold N* and vocab, vary depth/width aspect ratio (n_layer),.
    # 17/06/2026 found less layers improve validation loss (likely due to len(title) <<)
    "shape": Sweep(
        name="shape",
        axes={"n_layer": [2, 4, 6, 8, 10]},
        hold={"vocab_size": 64_000},
        resolve=solve_width(budget=96_000_000, head_dim=64),
        x="n_layer",
        tokens_per_step=4096,
    ),
}
