"""
MLM critic -- a small bidirectional encoder used as a *secondary* eval signal.

Trained from scratch on the real HN titles with a masked-LM objective, it scores any text by its
pseudo-log-likelihood (Salazar et al. 2019, arXiv:1910.14659): mask each token in turn and sum
-log p(token | rest). Lower nats/char = more "title-like". It is a learned, reference-free critic
of *free-running* generations -- it catches drift/degeneration that teacher-forced validation loss
is blind to. It NEVER touches the scored model, evaluate(), or the validation loss.

The critic has its own fixed tokenizer, so it scores decoded *text* and is invariant to whichever
tokenizer the model-under-test used (important: we sweep tokenizers). Train it once -> it is then a
frozen ruler for every model you compare.

Masking follows BERT (Devlin et al. 2018, arXiv:1810.04805) with the 80/10/10 split, but at a 30%
rate (raised from 15% because titles are short -> 15% leaves too few targets per title).

Train:   python3 critic.py train       # writes ./critic/{model.pt,tokenizer.json}
Score HN Model using eval_gen.py
"""
import json, math, sys
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer as HFTokenizer
from tqdm import tqdm

from train import get_titles
from model.tokenizer import train_tokenizer

CRITIC_DIR = "./critic"      # frozen artifact: model.pt + tokenizer.json
MASK_TOKEN = "[mask]"


@dataclass
class CriticConfig:
    vocab_size: int
    block_size: int = 32     # titles are short; 32 covers ~99% and avoids mostly-padding sequences
    n_layer: int = 8
    n_head: int = 6          # head_dim 64
    d_model: int = 384
    dropout: float = 0.1


class EncoderBlock(nn.Module):
    """Pre-norm transformer block with *bidirectional* self-attention (no causal mask)."""
    def __init__(self, cfg: CriticConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_head,
                                          dropout=cfg.dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model), nn.Dropout(cfg.dropout))

    def forward(self, x, key_padding_mask=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class Critic(nn.Module):
    def __init__(self, cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight          # tied
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ids, pad_id=None):
        T = ids.size(1)
        x = self.tok_emb(ids) + self.pos_emb[:, :T, :]
        kpm = ids.eq(pad_id) if pad_id is not None else None   # True = ignore (padding)
        for b in self.blocks:
            x = b(x, kpm)
        return self.head(self.ln_f(x))                  # (B, T, vocab)


def mlm_mask(ids, mask_id, vocab_size, pad_id, p=0.30):
    """BERT 80/10/10 corruption. Returns (corrupted ids, labels), labels=-100 where not scored.
    p is raised above BERT's 15% because titles are short -- 15% leaves only ~2 targets per title."""
    labels = ids.clone()
    prob = torch.full(ids.shape, p, device=ids.device)
    prob[ids == pad_id] = 0.0                                  # never mask padding
    masked = torch.bernoulli(prob).bool()
    labels[~masked] = -100                                     # only score masked positions
    r = torch.rand(ids.shape, device=ids.device)
    ids = ids.clone()
    ids[masked & (r < 0.8)] = mask_id                          # 80% -> [mask]
    rnd = torch.randint(0, vocab_size, ids.shape, device=ids.device)
    pick = masked & (r >= 0.8) & (r < 0.9)
    ids[pick] = rnd[pick]                                      # 10% -> random ; remaining 10% kept
    return ids, labels


def _encode_titles(titles, raw, eos_id, block_size, pad_id):
    """Each title -> [tokens..., <eos>] right-padded to block_size (truncate long ones)."""
    rows = []
    for t in titles:
        x = raw.encode(t).ids[:block_size - 1] + [eos_id]
        rows.append(x + [pad_id] * (block_size - len(x)))
    return torch.tensor(rows, dtype=torch.long)


def train_critic(vocab_size=8000, epochs=25, batch_size=128, lr=5e-4, seed=1337,
                 num_titles=100_000, val_frac=0.10, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)

    # train on TRAIN titles only -> val titles stay unseen, so they can serve as an unbiased
    # "real" anchor when reading critic scores later.
    titles = get_titles(num_titles, seed, val_frac)[0]
    raw = train_tokenizer("bpe", titles, vocab_size)           # specials: <pad> <eos> <unk>
    raw.add_special_tokens([MASK_TOKEN])                       # appended at the end of the vocab
    pad_id = raw.token_to_id("<pad>")
    eos_id = raw.token_to_id("<eos>")
    mask_id = raw.token_to_id(MASK_TOKEN)

    cfg = CriticConfig(vocab_size=raw.get_vocab_size())
    model = Critic(cfg).to(device)
    data = _encode_titles(titles, raw, eos_id, cfg.block_size, pad_id).to(device)
    print(f"critic: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params  "
          f"vocab={cfg.vocab_size}  titles={len(data)}  device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n_batches = math.ceil(len(data) / batch_size)
    total = epochs * n_batches
    warmup = max(1, int(0.05 * total))                          # short linear warmup, then cosine to 0
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    model.train()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(data), device=device)
        tot = 0.0
        for i in tqdm(range(0, len(data), batch_size), desc=f"critic epoch {ep}/{epochs}"):
            ids = data[perm[i:i + batch_size]]
            x, y = mlm_mask(ids, mask_id, cfg.vocab_size, pad_id)
            loss = F.cross_entropy(model(x, pad_id=pad_id).view(-1, cfg.vocab_size),
                                   y.view(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        print(f"  epoch {ep}: mlm_loss={tot / n_batches:.4f}")

    out = Path(CRITIC_DIR); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": asdict(cfg),
                "mask_id": mask_id, "pad_id": pad_id}, out / "model.pt")
    raw.save(str(out / "tokenizer.json"))
    print(f"saved critic -> {out}")


def load_critic(critic_dir=CRITIC_DIR, device="cpu"):
    ck = torch.load(Path(critic_dir) / "model.pt", map_location=device, weights_only=False)
    model = Critic(CriticConfig(**ck["config"]))
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    raw = HFTokenizer.from_file(str(Path(critic_dir) / "tokenizer.json"))
    return model, raw, ck["mask_id"], ck["pad_id"]


@torch.no_grad()
def pll(model, ids, mask_id, pad_id):
    """Pseudo-log-likelihood (nats) of a 1-D token sequence: mask each position once and sum
    -log p(true token | rest). All n masked variants are batched into a single forward pass."""
    n = ids.size(0)
    batch = ids.unsqueeze(0).repeat(n, 1)
    d = torch.arange(n, device=ids.device)
    batch[d, d] = mask_id                                      # row i masks position i
    logp = model(batch, pad_id=pad_id)[d, d].log_softmax(-1)   # (n, vocab) at the masked slots
    return -logp[d, ids].sum().item()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_critic()
    else:
        sys.exit("usage: python3 critic.py train")
