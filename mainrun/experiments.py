"""
Experiment configs

Each entry is a Sweep (see bench.py for the schema). To add a benchmark, add an entry;
to run it:  `python3 bench.py run <key>`  /  re-plot:  `python3 bench.py plot <key>`.

Conventions used below:
  * solve_width(budget=...) derives d_model + n_head for each point. `budget` is a fixed
    param count (iso-param sweeps) or the name of a swept axis like "target_N" (size/find-N).
  * axes with >1 key form a cartesian grid (e.g. find_n = vocab x target_N).
  * a one-off comparison (MHA vs MLA, dropout on/off, ...) is just a short axis -- no new file.
"""
from bench import Sweep, solve_width, heads_for


# one-flag-at-a-time ablation: map a 'variant' selector -> the single arch override it flips.
# 'baseline' = all arch flags default (a plain model + WSD/muon from `hold`); each other variant
# turns on exactly ONE of our additions -> clean per-idea attribution against the same baseline.
_ABLATION = {
    "baseline": {},                          # everything default (layernorm, gelu, learned pos, no qk-norm, bias on)
    "rmsnorm":  {"norm_type": "rmsnorm"},
    "swiglu":   {"mlp_type": "swiglu"},
    "rope":     {"pos_type": "rope"},
    "qk_norm":  {"qk_norm": True},
    "bias_off": {"bias": "off"},
}
def ablate_arch(p):
    return _ABLATION[p["variant"]]


EXPERIMENTS = {
    # baseline:
    "baseline": Sweep(
        name="baseline",
        axes={"vocab_size": [16_000]},
    ),

    # --- batch-size sweep (should have come first): the rest of the campaign pins tokens/step=4096
    # -> batch_size=32 (4096/128), overriding the dataclass default of 64. Confirm 32 is the right
    # pick before everything downstream inherits it. batch x step-count are coupled: halving batch
    # doubles max_steps over the fixed 7 epochs, and optimal lr scales with batch -- so we co-sweep
    # lr and compare each batch AT ITS OWN BEST lr (one curve per batch), rather than at a shared lr.
    # Run before `optim` so the optimiser sweep inherits the winning batch. adamw used as the
    # stage default (optimiser not chosen yet); whatever batch wins, set it as the new default.
    # 19/06/2026
    "batch": Sweep(
        name="batch",
        axes={"batch_size": [16, 32, 64],
              "lr": [1e-4, 3e-4, 1e-3, 3e-3]},
        hold={"optim_alg": "adamw", "optim_type": "cosine", "n_layer": 6},
        x="lr", group="batch_size",
        # tokens_per_step is ignored here (batch_size is explicit) -> tokens/step = batch*block varies.
    ),

    # gradient-descent baseline
    # 17/06/2026: Plain SGD has potential to underfit transformers badly, benchmark vs. adamw
    # 18/06/2026: added muon (hybrid: Muon on 2D transformer matrices + AdamW on embeddings/head/
    #             norms/biases). Optimizers have different lr regimes (sgd/adamw lower, muon ~1e-2+),
    #             so the grid spans both ends; compare each optimizer at its own best lr. lr drives
    #             Muon's matrices (axis lr); the AdamW-aux group runs at args.lr_hybird (default 3e-4).
    #             Found: best: optim_alg=muonhybrid, lr=0.01  val=1.2767
    "optim": Sweep(
        name="optim_baseline",
        axes={"optim_alg": ["sgd", "adamw", "muonhybrid"],
              "lr": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]},
        x="lr", group="optim_alg",
        tokens_per_step=4096,
    ),

    # --- WSD decay sweep: optim_type='wsd' (warmup-stable-decay) on the muon-hybrid winner.
    # The optim sweep used cosine @ muonhybrid lr=1e-2 -> val=1.2767; this holds the SAME config
    # (so 1.2767 is the cosine baseline to beat) and asks how much a stable plateau + late decay
    # squeezes out. x = decay_frac (fraction of steps spent decaying), one curve per decay shape.
    # warmup held at 0.05; sweep warmup separately if decay looks promising.
    # 18/06/2026
    "decay": Sweep(
        name="wsd_decay",
        axes={"decay_frac": [0.1, 0.2, 0.3, 0.4, 0.6],
              "decay_type": ["linear", "cosine", "sqrt"]},
        hold={
            "optim_type": "wsd",
            "optim_alg": "muonhybrid",
            "lr": 1e-2,
            "warmup_frac": 0.05,
            "n_layer": 6,
        },
        x="decay_frac", group="decay_type",
        tokens_per_step=4096,
    ),

    # --- architecture ablation: one flag at a time vs a shared baseline.
    # Baseline = a default model + the winning optimiser/schedule (muonhybrid, wsd decay_frac=0.1 sqrt);
    # tokenizer + size + all arch flags left at default. Each non-baseline variant flips exactly ONE of
    # our additions (rmsnorm / swiglu / rope / qk_norm / bias-off) so its delta is cleanly attributable.
    # Keep what helps, then re-check the surviving bundle together (interactions).
    # 19/06/2026
    "ablation": Sweep(
        name="arch_ablation",
        axes={"variant": ["baseline", "rmsnorm", "swiglu", "rope", "qk_norm", "bias_off"]},
        hold={
            "optim_type": "wsd",
            "optim_alg": "muonhybrid",
            "lr": 1e-2,
            "warmup_frac": 0.05,
            "decay_frac": 0.1,
            "decay_type": "sqrt",
            "n_layer": 6,
        },
        resolve=ablate_arch,          # variant -> single arch override (variant itself isn't a HP field)
        x="variant",
        tokens_per_step=4096,
    ),

    # --- tokenizer benchmark: fix the ~32M param budget, identify strongest BPE vocab size.
    # 16/06/2026: found 64k to be strongest (BPE only)
    # 17/06/2026: added tokenizer axis -> grid of {bpe,unigram,wordpiece} x vocab_size.
    #             (every (tok, vocab) cell is resized to ~32M params (close to default) -> a fair head-to-head.)
    # 17/06/2026: added superbpe (two-stage; transition_ratio held at the 0.75 default for this grid).
    # 17/06/2026: move optimiser sweep to be prior, found adamw = lr 3e-4 to be suitable
    "tok_isoparam": Sweep(
        name="tok_isoparam_32M",
        axes={"token_type": ["bpe", "unigram", "wordpiece", "superbpe"],
              "vocab_size": [8_000, 16_000, 32_000, 64_000, 96_000]},
        hold={
            "n_layer": 6, 
            "optim_alg": "muonhybrid", 
            "optim_type": "wsd",
            "lr": 1e-2,
            "warmup_frac": 0.05,
            "decay_frac": 0.1,
            "decay_type": "sqrt",
            "norm_type": "rmsnorm",
            "pos_type": "rope",
        },
        resolve=solve_width(budget=32_000_000, head_dim=64),
        x="vocab_size", group="token_type",
        tokens_per_step=4096,
    ),

    # --- joint architecture search: size and shape were each 1-D (size = width@L6, shape = depth@fixed-N),
    # so neither saw the depth x width interaction. shape went monotonically deeper -> maybe bigger N WITH
    # more layers wins. Here params are NOT pinned: we grid depth x width directly and let N (total, incl.
    # embedding) fall out onto the x-axis. n_head is derived (head_dim~64). N spans ~35M-190M @ 64k vocab.
    # Read it as val vs N, one curve per depth -> does the deeper curve keep dropping as N grows?
    # 18/06/2026
    "arch": Sweep(
        name="arch",
        axes={"n_layer": [6, 8, 12, 16, 20],
              "d_model": [384, 512, 640, 768]},   # multiples of 64 -> exact head_dim=64
        hold={
            # "vocab_size": 16_000, 
            # "token_type": "unigram", 
            "optim_alg": "muonhybrid", 
            "lr": 1e-2,
            "optim_type": "wsd",
            "warmup_frac": 0.05,
            "decay_frac": 0.1,
            "decay_type": "sqrt",
            "norm_type": "rmsnorm",
            "pos_type": "rope",
        },
        resolve=heads_for(head_dim=64),           # only sets n_head; d_model/n_layer come from the axes
        x="N", group="n_layer",
        tokens_per_step=4096,
    ),

    # --- regularisation sweep: at the winning config, tune weight_decay x dropout.
    "reg": Sweep(
        name="reg",
        axes={"weight_decay": [0.0, 0.01, 0.1],
              "dropout": [0.0, 0.1, 0.2]},
        # hold={"vocab_size": 64_000, "n_layer": ___,         # fill from size / shape results
        #       "optim_type": ___, "lr": ___},                # fill from optim results
        # resolve=solve_width(budget=___, head_dim=64),       # fill N* from size results
        x="weight_decay", group="dropout",
        tokens_per_step=4096,
    ),
}
