import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import math

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    d_model: int
    dropout: float
    norm_type: str = 'layernorm'   # defaults = original arch, so old checkpoints reload unchanged
    mlp_type:  str = 'gelu'
    pos_type:  str = 'learned'
    qk_norm:   bool = False
    bias:      str = 'default'  # 'default' = original (nn.Linear + LayerNorm biases on); 'off' = no bias anywhere


def _use_bias(cfg) -> bool:
    """'default' -> original Linear/LayerNorm biases; 'off' -> bias removed everywhere it applies
    (qkv, proj, MLP, SwiGLU, and the LayerNorm affine biases). The output head is bias-free either way."""
    return cfg.bias == 'default'

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_head   = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=_use_bias(cfg))
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=_use_bias(cfg))
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        self.dropout = cfg.dropout

        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)   # normalise each head's vector (last dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x: torch.Tensor, cos=None, sin=None):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]   # each (B, nh, T, hd)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if cos is not None:                       # pos_type == 'rope'
            q, k = apply_rope(q, k, cos, sin)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels ref: https://github.com/karpathy/nanoGPT/blob/master/model.py
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

# SwiGlu https://arxiv.org/abs/2002.05202
class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(8/3 * cfg.d_model)          # 8/3 keeps param count == 4*d GELU MLP
        hidden = 64 * ((hidden + 63) // 64)      # round to multiple of 64
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=_use_bias(cfg))   # gate
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=_use_bias(cfg))   # up
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=_use_bias(cfg))   # down
        self.drop = nn.Dropout(cfg.dropout)
    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=_use_bias(cfg)),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model, bias=_use_bias(cfg)),
            nn.Dropout(cfg.dropout),
        )
    def forward(self, x): return self.net(x)

# RMSNorm for efficiency https://arxiv.org/abs/1910.07467
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.norm_type == 'layernorm':
            self.ln1 = nn.LayerNorm(cfg.d_model, bias=_use_bias(cfg))
            self.ln2 = nn.LayerNorm(cfg.d_model, bias=_use_bias(cfg))
        elif cfg.norm_type == 'rmsnorm':
            self.ln1 = RMSNorm(cfg.d_model)
            self.ln2 = RMSNorm(cfg.d_model)
        else:
            raise ValueError(f"Unknown norm_type: {cfg.norm_type}")
        self.attn = CausalSelfAttention(cfg)
        if cfg.mlp_type == 'gelu':
            self.mlp  = MLP(cfg)
        elif cfg.mlp_type == 'swiglu':
            self.mlp  = SwiGLU(cfg)
        else:
            raise ValueError(f"Unknown mlp_type: {cfg.mlp_type}")
    def forward(self, x, cos=None, sin=None):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x

# RoPE https://arxiv.org/abs/2104.09864
def build_rope_cache(seq_len, head_dim, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)              # (T, hd/2)
    emb = torch.cat((freqs, freqs), dim=-1)       # (T, hd)
    return emb.cos(), emb.sin()                   # each (T, hd)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):                   # q,k: (B, nh, T, hd)
    cos, sin = cos[None, None], sin[None, None]   # -> (1,1,T,hd)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos_type == 'learned':
            self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        elif cfg.pos_type == 'rope':
            self.pos_emb = None
            head_dim = cfg.d_model // cfg.n_head
            assert head_dim % 2 == 0, "RoPE needs an even head_dim"
            cos, sin = build_rope_cache(cfg.block_size, head_dim)
            self.register_buffer("rope_cos", cos, persistent=False)  # non-persistent: moves with .to(device), not saved
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            raise ValueError(f"Unknown pos_type: {cfg.pos_type}")
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        if cfg.norm_type == 'layernorm':
            self.ln_f = nn.LayerNorm(cfg.d_model, bias=_use_bias(cfg))
        elif cfg.norm_type == 'rmsnorm':
            self.ln_f = RMSNorm(cfg.d_model)
        else:
            raise ValueError(f"Unknown norm_type: {cfg.norm_type}")
        self.head      = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.size()
        x = self.token_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb[:, :T, :]
        x = self.drop(x)
        cos = self.rope_cos[:T] if self.cfg.pos_type == 'rope' else None
        sin = self.rope_sin[:T] if self.cfg.pos_type == 'rope' else None
        for block in self.blocks: x = block(x, cos, sin)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
        return logits, loss

    # adapted from https://github.com/karpathy/nanoGPT/blob/master/model.py
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        If eos_id is given, generation stops once every sequence in the batch has emitted it
        (titles are <eos>-delimited, so this ends a sample at its title boundary).
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            # stop early once the whole batch has produced <eos>
            if eos_id is not None and (idx_next == eos_id).all():
                break

        return idx