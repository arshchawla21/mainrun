"""
GPT v2 -- the Phase A backbone, locked.

Phase A selected RMSNorm + RoPE + per-title masking (GELU MLP, biases kept, no QK-norm). v2 hardwires
those, so the architecture is fixed and the code is branch-free. This gives a clean base to ablate
*new* (Phase B) ideas -- e.g. alternative attention, value residuals -- without the per-flag if/else
that gpt.py carries from the Phase A search.

Training-side knobs still come from config (lr/block_size/dropout/n_layer/d_model/vocab/title_masking);
the Phase A architecture flags (norm_type, mlp_type, pos_type, qk_norm, bias) are ignored here. Shared
leaf modules are imported from gpt.py so there is a single source of truth for each component.

Select it from train.py with `gpt_version='v2'`.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from model.config import GPTConfig
from model.gpt import RMSNorm, MLP, RotaryEmbedding, apply_rope


# ============================ attention variants (Phase B) ============================
# Uniform interface: forward(x, cos, sin, mask, v_theta) -> (y, v). Every variant returns its value tensor
# so GPT2 can hand layer-theta's v down as v_theta (used only by value_residual). `mask` is a FlexAttention
# BlockMask (flex path), an additive float mask (differential -> SDPA), or None (causal SDPA).

class Attention(nn.Module):
    """Baseline multi-head attention: RoPE + per-title mask.  # Vaswani et al. 2017, arXiv:1706.03762"""
    def __init__(self, cfg: GPTConfig, n_head):
        super().__init__()
        assert cfg.d_model % n_head == 0
        self.n_head   = n_head
        self.head_dim = cfg.d_model // n_head
        self.dropout  = cfg.dropout
        self.qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def _qkv(self, x):
        B, T, _ = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
        return qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]   # each (B, nh, T, hd)

    def _attn(self, q, k, v, mask):
        # mask is None (causal) or an additive per-title float mask (B,1,T,T). SDPA throughout so every
        # variant works -- incl. single_head (head_dim=d_model) and differential (q/k head_dim != v).
        if mask is None:
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    def forward(self, x, cos, sin, mask=None, v0=None):
        B, T, C = x.size()
        q, k, v = self._qkv(x)
        q, k = apply_rope(q, k, cos, sin)
        y = self._attn(q, k, v, mask).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y)), v


class ValueResidualAttention(Attention):
    """v <- (1-λ)v + λ.vtheta, mixing in the first layer's values (a value shortcut across depth).
    # Zhou et al. 2024, arXiv:2410.17897"""
    def __init__(self, cfg: GPTConfig, n_head):
        super().__init__(cfg, n_head)
        self.lam = nn.Parameter(torch.zeros(1))        # sigmoid(0)=0.5

    def forward(self, x, cos, sin, mask=None, v0=None):
        B, T, C = x.size()
        q, k, v = self._qkv(x)
        q, k = apply_rope(q, k, cos, sin)
        if v0 is not None:                             # layer 0 has v0=None -> returns its raw v
            g = torch.sigmoid(self.lam)
            v = (1 - g) * v + g * v0
        y = self._attn(q, k, v, mask).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y)), v


class OutputGatedAttention(Attention):
    """Per-head sigmoid gate on the attention output.  # Qiu et al. 2025, arXiv:2505.06708"""
    def __init__(self, cfg: GPTConfig, n_head):
        super().__init__(cfg, n_head)
        self.gate = nn.Linear(cfg.d_model, n_head)

    def forward(self, x, cos, sin, mask=None, v0=None):
        B, T, C = x.size()
        q, k, v = self._qkv(x)
        q, k = apply_rope(q, k, cos, sin)
        y = self._attn(q, k, v, mask)                  # (B, nh, T, hd)
        g = torch.sigmoid(self.gate(x)).permute(0, 2, 1).unsqueeze(-1)   # (B, nh, T, 1)
        y = (y * g).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y)), v


class DifferentialAttention(Attention):
    """Two softmax maps subtracted: (A1 - λ·A2)·V, cancelling common-mode attention noise. Each head's
    q/k split into two hd/2 halves (iso-param), each rope'd; runs via SDPA (additive mask) so v may keep
    full head_dim.  # Ye et al. 2024, arXiv:2410.05258 (Differential Transformer)"""
    def __init__(self, cfg: GPTConfig, n_head):
        super().__init__(cfg, n_head)
        assert self.head_dim % 2 == 0, "differential splits head_dim in two"
        self.rope_half = RotaryEmbedding(self.head_dim // 2, cfg.block_size)
        self.lam = nn.Parameter(torch.tensor(0.5))
        self.out_norm = RMSNorm(self.head_dim)

    def forward(self, x, cos, sin, mask=None, v0=None):
        B, T, C = x.size()
        q, k, v = self._qkv(x)                         # (B, nh, T, hd)
        c2, s2 = self.rope_half(T)                     # (T, hd/2)
        q1, q2 = q.chunk(2, -1); k1, k2 = k.chunk(2, -1)
        q1, k1 = apply_rope(q1, k1, c2, s2)
        q2, k2 = apply_rope(q2, k2, c2, s2)
        y = self._attn(q1, k1, v, mask) - self.lam * self._attn(q2, k2, v, mask)   # (B, nh, T, hd)
        y = self.out_norm(y).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y)), v


class GatedValueResidualAttention(Attention):
    """Value residual + output gate together (orthogonal: one edits v pre-attn, one gates y post-attn).
    # combines [ValueResidual] arXiv:2410.17897 + [GatedAttn] arXiv:2505.06708"""
    def __init__(self, cfg: GPTConfig, n_head):
        super().__init__(cfg, n_head)
        self.lam  = nn.Parameter(torch.zeros(1))
        self.gate = nn.Linear(cfg.d_model, n_head)

    def forward(self, x, cos, sin, mask=None, v0=None):
        B, T, C = x.size()
        q, k, v = self._qkv(x)
        q, k = apply_rope(q, k, cos, sin)
        if v0 is not None:
            g = torch.sigmoid(self.lam)
            v = (1 - g) * v + g * v0
        y = self._attn(q, k, v, mask)
        gate = torch.sigmoid(self.gate(x)).permute(0, 2, 1).unsqueeze(-1)
        y = (y * gate).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y)), v


ATTENTIONS = {
    'mha':                  Attention,
    'value_residual':       ValueResidualAttention,
    'output_gated':         OutputGatedAttention,
    'differential':         DifferentialAttention,
    'gated_value_residual': GatedValueResidualAttention,
}


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, attn_cls, n_head):
        super().__init__()
        self.ln1  = RMSNorm(cfg.d_model)
        self.ln2  = RMSNorm(cfg.d_model)
        self.attn = attn_cls(cfg, n_head)
        self.mlp  = MLP(cfg)
    def forward(self, x, cos, sin, mask=None, v0=None):
        a, v = self.attn(self.ln1(x), cos, sin, mask, v0)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, v


class GPT2(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        single = cfg.attn_type == 'single_head'
        self.n_head = 1 if single else cfg.n_head          # single_head -> one head over the full d_model
        attn_cls = ATTENTIONS['mha' if single else cfg.attn_type]

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope      = RotaryEmbedding(cfg.d_model // self.n_head, cfg.block_size)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([Block(cfg, attn_cls, self.n_head) for _ in range(cfg.n_layer)])
        self.ln_f      = RMSNorm(cfg.d_model)
        self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.head.weight = self.token_emb.weight         # tied input/output embeddings

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        T = idx.size(1)
        x = self.drop(self.token_emb(idx))
        cos, sin = self.rope(T)
        mask = self._mask(idx) if self.cfg.title_masking else None
        v0 = None
        for i, block in enumerate(self.blocks):
            x, v = block(x, cos, sin, mask, v0)
            if i == 0:
                v0 = v                                   # first layer's values, for value_residual
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def _mask(self, idx):
        """Additive per-title mask (B,1,T,T): 0 where attention is allowed (causal AND same <eos>-title),
        -inf otherwise. Built from the tokens each forward, so eval/generate stay consistent."""
        B, T = idx.size()
        doc_ids = (idx == self.cfg.eos_id).cumsum(-1)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=idx.device))
        allow = causal[None] & (doc_ids[:, :, None] == doc_ids[:, None, :])          # (B, T, T)
        return torch.where(allow[:, None], 0.0, float('-inf'))                       # (B, 1, T, T)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        """Autoregressive sampling (see gpt.py GPT.generate for the conventions)."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_id is not None and (idx_next == eos_id).all():
                break
        return idx
