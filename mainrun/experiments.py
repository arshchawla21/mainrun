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


# Phase B rollback ablation: from the best model, flip ONE Phase-A-rejected choice back on, to check
# coordinate descent still holds now that the rest of the model has changed (one override per variant).
# Arch rollbacks stay at the best d512. Large-vocab tokenizers OOM at d512 under R-Drop (64k/96k
# embeddings), so they + a matched unigram-16k reference run at d384/h6 -> compare those to unigram_d384.
_ROLLBACK = {
    "best":          {},                                                   # the current best, d512 (reference)
    "bpe":        {"token_type": "bpe",},
    "wordpiece":  {"token_type": "wordpiece"},
    "superbpe":   {"token_type": "superbpe"},
    "swiglu":        {"mlp_type": "swiglu"},
    "qk_norm":       {"qk_norm": True},
    "bias_off":      {"bias": "off"},
}
def rollback(p):
    return _ROLLBACK[p["variant"]]


EXPERIMENTS = {
    # baseline:
    "baseline": Sweep(
        name="0_baseline",
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
        name="1_optim",
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
        name="2_wsd_decay",
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
        name="3_arch",
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
    # 19/06/2026: run with latest architecture + muonhybrid opt
    "tok": Sweep(
        name="4_tokenizer",
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

    # --- joint architecture/shape search: size and shape were each 1-D (size = width@L6, shape = depth@fixed-N),
    # so neither saw the depth x width interaction. shape went monotonically deeper -> maybe bigger N WITH
    # more layers wins. Here params are NOT pinned: we grid depth x width directly and let N (total, incl.
    # embedding) fall out onto the x-axis. n_head is derived (head_dim~64). N spans ~35M-190M @ 64k vocab.
    # Read it as val vs N, one curve per depth -> does the deeper curve keep dropping as N grows?
    # 18/06/2026: 
    # 19/06/2026: found layers12, d384, heads6
    "shape": Sweep(
        name="5_shape",
        axes={"n_layer": [6, 8, 12, 16, 20],
              "d_model": [256, 384, 512, 640, 768]},   # multiples of 64 -> exact head_dim=64
        hold={
            "vocab_size": 16_000, 
            "token_type": "unigram", 
            "optim_alg": "muonhybrid", 
            "optim_type": "wsd",
            "lr": 1e-2,
            "warmup_frac": 0.05,
            "decay_frac": 0.1,
            "decay_type": "sqrt",
            "norm_type": "rmsnorm",
            "pos_type": "rope",
        },
        resolve=heads_for(head_dim=64), # only sets n_head; d_model/n_layer come from the axes
        x="N", group="n_layer",
        tokens_per_step=4096,
    ),

    # --- title masking ablation: FlexAttention block-diagonal mask confining attention within each
    # <eos>-delimited title vs the current full-causal baseline. Both legs are pos_type='rope'
    # (masking asserts relative positions). eos_id is derived in train.py, so nothing to set here.
    # After this, re-run `block` with the winner -> if the block curve flattens, the earlier
    # larger-is-better was cross-boundary leakage (see changes.md).
    # 19/06/2026
    "masking": Sweep(
        name="6_masking",
        axes={"title_masking": [False, True]},
        hold={
            "vocab_size": 16_000,
            "token_type": "unigram",
            "optim_alg": "muonhybrid",
            "optim_type": "wsd",
            "lr": 1e-2,
            "warmup_frac": 0.05,
            "decay_frac": 0.1,
            "decay_type": "sqrt",
            "norm_type": "rmsnorm",
            "pos_type": "rope",
            "n_layer": 12,
            "d_model": 384,
            "n_head": 6
        },
        x="title_masking",
        tokens_per_step=4096,
    ),

    # --- tokens-per-step sweep at FIXED block_size=128 (= a batch sweep, framed as tokens/step).
    # block held at 128, so batch_size IS tokens/step (= batch*128): batch 8/16/32/64 -> ts 1024/2048/
    # 4096/8192. Fewer tokens/step -> more optimiser steps over the fixed 7 epochs (and noisier grads).
    # lr fixed at 0.01 (established best); tokens_per_step NOT pinned so batch_size drives it.
    # 19/06/2026
    "tps": Sweep(
        name="7_tokens_per_step",
        axes={"batch_size": [8, 16, 32, 64]},   # @block128 -> tokens/step 1024, 2048, 4096, 8192
        hold={
            "block_size": 128,
            "vocab_size": 16_000,
            "token_type": "unigram",
            "optim_alg": "muonhybrid",
            "optim_type": "wsd",
            "lr": 1e-2,
            "warmup_frac": 0.05,
            "decay_frac": 0.1,
            "decay_type": "sqrt",
            "norm_type": "rmsnorm",
            "pos_type": "rope",
            "n_layer": 12,
            "d_model": 384,
            "n_head": 6,
            "title_masking": True,
        },
        x="batch_size",
    ),

    # --- block at fixed tokens per step
    "block": Sweep(
        name="8_block",
        axes={"block_size": [16, 32, 64, 128, 256, 512]},
        hold={
            "vocab_size": 16_000,
            "token_type": "unigram",
            "optim_alg": "muonhybrid",
            "optim_type": "wsd",
            "lr": 1e-2,
            "warmup_frac": 0.05,
            "decay_frac": 0.1,
            "decay_type": "sqrt",
            "norm_type": "rmsnorm",
            "pos_type": "rope",
            "n_layer": 12,
            "d_model": 384,
            "n_head": 6,
            "title_masking": True
        },
        x="block_size",
        tokens_per_step=4096,
    ),

    # --- regularisation sweep: at the winning config, tune dropout.
    "reg": Sweep(
        name="9_reg",
        axes={"dropout": [0.0, 0.1, 0.2]},
        hold={
            "vocab_size": 16_000,
            "token_type": "unigram",
            "optim_alg": "muonhybrid",
            "optim_type": "wsd",
            "lr": 1e-2,
            "warmup_frac": 0.05,
            "decay_frac": 0.1,
            "decay_type": "sqrt",
            "norm_type": "rmsnorm",
            "pos_type": "rope",
            "n_layer": 12,
            "d_model": 384,
            "n_head": 6,
            "title_masking": True,
            "block_size": 256
        },
        x="dropout",
        tokens_per_step=4096,
    ),

    # --- Phase B attention ablation (gpt_v2). Swap the whole attention component at the Phase A
    # optimal config: baseline MHA, single-head (lower bound), value-residual, output-gated, and
    # differential. All iso-param. Capacity-limited model -> we bet on capacity/inductive-bias
    # variants, not memory-savers (GQA/MLA dropped). See report "Alternative Attention Design".
    # 20/06/2026
    "attn": Sweep(
        name="10_attn",
        axes={"attn_type": ["mha", "single_head", "value_residual", "output_gated", "differential",
                            "gated_value_residual"]},
        hold={
            "gpt_v2": True,              # required: v1 ignores attn_type
            "vocab_size": 16_000,
            "token_type": "unigram",
            "optim_alg": "muonhybrid",
            "optim_type": "wsd",
            "lr": 1e-2,
            "warmup_frac": 0.05,
            "decay_frac": 0.1,
            "decay_type": "sqrt",
            "title_masking": True,
            "n_layer": 12,
            "d_model": 384,
            "n_head": 6,
            "block_size": 256,
            "dropout": 0.2,
        },
        x="attn_type",
        tokens_per_step=4096,
    ),

    # --- Phase B cross-layer residual paths (gpt_v2), on the output_gated backbone (attention winner).
    # All extra information paths beyond the per-block residual: unet (symmetric encoder-decoder skips),
    # embedding_shortcut (feed token embeddings into deep layers), layerscale (learned per-channel
    # residual scale). All gated/iso-param. See report "Cross-layer residual paths".
    # 20/06/2026
    "residual": Sweep(
        name="11_residual",
        axes={"residual": ["none", "unet", "embedding_shortcut", "layerscale"]},
        hold={
            "gpt_v2": True,
            "attn_type": "output_gated",     # carry the attention winner forward
            "vocab_size": 16_000,
            "token_type": "unigram",
            "optim_alg": "muonhybrid",
            "optim_type": "wsd",
            "lr": 1e-2,
            "warmup_frac": 0.05,
            "decay_frac": 0.1,
            "decay_type": "sqrt",
            "title_masking": True,
            "n_layer": 12,
            "d_model": 384,
            "n_head": 6,
            "block_size": 256,
            "dropout": 0.2,
        },
        x="residual",
        tokens_per_step=4096,
    ),

    # --- Phase B per-head attention temperature (gpt_v2), on the output_gated backbone. A learned
    # per-(layer,head) scalar on q -> exp(g_h) scales the q.k scores, so each head picks its own softmax
    # sharpness (zero-init = no-op). Idea informed by our own attention ablation: per-head dials on
    # attention win (output_gated, value_residual), so we add one on a new axis (the distribution).
    # See report "Per-head attention temperature".
    # 21/06/2026
    "attn_temp": Sweep(
        name="12_attn_temp",
        axes={"attn_temp": [False, True]},
        hold={
            "gpt_v2": True,
            "attn_type": "output_gated",
            "vocab_size": 16_000,
            "token_type": "unigram",
            "optim_alg": "muonhybrid",
            "optim_type": "wsd",
            "lr": 1e-2,
            "warmup_frac": 0.05,
            "decay_frac": 0.1,
            "decay_type": "sqrt",
            "title_masking": True,
            "n_layer": 12,
            "d_model": 384,
            "n_head": 6,
            "block_size": 256,
            "dropout": 0.2,
            "residual": "layerscale"
        },
        x="attn_temp",
        tokens_per_step=4096,
    ),

    # --- aux learning-rate sweep (the second, untuned knob of our biggest win). muonhybrid drives the
    # 2D matrices at lr=1e-2 but the AdamW-aux group (embeddings, tied head, norms, biases) runs at
    # lr_hybird=3e-4 -- a 33x gap we never tuned. The embedding diagnostic (embed_diag.py) ruled out
    # rare-token under-training (tied head -> dense gradients), so this is a balance/ratio question, not
    # a rescue; NorMuon's tuned setup runs embeddings/output ABOVE hidden, hinting 3e-4 may be too low.
    # 21/06/2026
    "aux_lr": Sweep(
        name="13_aux_lr",
        axes={"lr_hybird": [3e-4, 1e-3, 3e-3, 1e-2]},
        hold={
            "gpt_v2": True,
            "attn_type": "output_gated",
            "vocab_size": 16_000,
            "token_type": "unigram",
            "optim_alg": "muonhybrid",
            "optim_type": "wsd",
            "lr": 1e-2,
            "warmup_frac": 0.05,
            "decay_frac": 0.1,
            "decay_type": "sqrt",
            "title_masking": True,
            "n_layer": 12,
            "d_model": 384,
            "n_head": 6,
            "block_size": 256,
            "dropout": 0.2,
        },
        x="lr_hybird",
        tokens_per_step=4096,
    ),

    # --- R-Drop x size frontier ---
    # 22/06/2026
    "rdrop_size": Sweep(
        name="13_rdrop_size",
        axes={"d_model": [384, 512, 640], "rdrop": [0.0, 2.0, 4.0]},
        hold={
            "gpt_v2": True,
            "attn_type": "output_gated",
            "residual": "layerscale",
            "amp": True,
            "vocab_size": 16_000,
            "token_type": "unigram",
            "optim_alg": "muonhybrid",
            "optim_type": "wsd",
            "lr": 1e-2,
            "warmup_frac": 0.05,
            "decay_frac": 0.1,
            "decay_type": "sqrt",
            "title_masking": True,
            "n_layer": 12,
            "block_size": 256,
            "dropout": 0.2,
        },
        resolve=heads_for(64),        # n_head = d_model/64 (head_dim 64)
        x="d_model", group="rdrop",
        tokens_per_step=4096,
    ),

    # --- rollback ablation: challenge coordinate descent on the FINAL Phase-B model
    "rollback": Sweep(
        name="14_rollback",
        axes={"variant": ["best", "swiglu", "qk_norm", "bias_off"]},
        hold={
            "gpt_v2": True, "attn_type": "output_gated", "residual": "layerscale",
            "rdrop": 2.0, "amp": True,
            "vocab_size": 16_000, "token_type": "unigram",
            "n_layer": 12, "d_model": 512, "n_head": 8, "block_size": 256, "dropout": 0.2,
            "optim_alg": "muonhybrid", "optim_type": "wsd", "lr": 1e-2, "lr_hybird": 3e-4,
            "warmup_frac": 0.05, "decay_frac": 0.1, "decay_type": "sqrt",
            "title_masking": True, "weight_decay": 0.01,
        },
        resolve=rollback,             # variant -> the single Phase-A choice it flips back on
        x="variant",
        tokens_per_step=4096,
    ),

    # --- Muon-group weight decay: the anti-overfit lever.
    # 23/06/2026
    "muon_wd": Sweep(
        name="15_muon_wd",
        axes={"muon_weight_decay": [0.0, 0.03, 0.05, 0.1, 0.15, 0.3]},
        hold={
            "gpt_v2": True, "attn_type": "output_gated", "residual": "layerscale",
            "rdrop": 2.0, "amp": True,
            "vocab_size": 16_000, "token_type": "unigram",
            "n_layer": 12, "d_model": 512, "n_head": 8, "block_size": 256, "dropout": 0.2,
            "optim_alg": "muonhybrid", "optim_type": "wsd", "lr": 1e-2, "lr_hybird": 3e-4,
            "warmup_frac": 0.05, "decay_frac": 0.1, "decay_type": "sqrt",
            "title_masking": True, "weight_decay": 0.01, "bias": "off"
        },
        x="muon_weight_decay",
        tokens_per_step=4096,
    ),
}
