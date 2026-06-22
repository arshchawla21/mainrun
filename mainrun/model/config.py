from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Shared by both model variants. gpt.py (v1) reads every field as an ablation switch; gpt_v2.py
    hardwires the Phase A winners and ignores the architecture flags (norm/mlp/pos/qk_norm/bias),
    using only the dims + dropout + title_masking/eos_id."""
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float
    norm_type: str = 'layernorm'   # v1 only; v2 hardwires RMSNorm
    mlp_type:  str = 'gelu'        # v1 only; v2 hardwires GELU MLP
    pos_type:  str = 'learned'     # v1 only; v2 hardwires RoPE
    qk_norm:   bool = False        # v1 only; v2 omits it
    bias:      str = 'default'     # v1 only; v2 keeps biases ('default')
    title_masking: bool = False    # both: attention confined within each <eos>-delimited title (needs rope)
    eos_id:    int = -1            # token id of <eos>; only read when title_masking
    attn_type: str = 'mha'        # v2 Phase B: mha | single_head | value_residual | output_gated | differential
    residual:  str = 'none'       # v2 Phase B cross-layer paths: none | unet | embedding_shortcut | layerscale
    attn_temp: bool = False       # v2 Phase B: per-(layer,head) learned attention temperature (zero-init no-op)
    label_smoothing: float = 0.0  # training-loss only: soften one-hot targets (regulariser); eval stays hard CE
    mtp:       int = 0            # v2 Phase B: # extra multi-token-prediction heads (0=off; k -> predict 2..k+1 ahead)
    mtp_type:  str = 'head'       # 'head' = vocab projection on shared rep | 'module' = dedicated transformer layer (decoupled)
    value_emb: int = 0            # v2 Phase B: # token-id value-embedding tables added into v (U-net pattern; 0=off)
