"""
Secondary eval: score a run's autoregressive generations with the MLM critic (critic.py).

Generate N titles from the model in <run_dir> (primed with <eos>, exactly as sample.py does), score
each by the critic's pseudo-log-likelihood, and write the mean to <run_dir>/eval.log. This is a
secondary signal only -- it does NOT touch evaluate() or the scored validation loss; it stress-tests
autoregressive generation quality, which teacher-forced val loss cannot see.

Run with:  python3 eval_gen.py <run_dir> [N]

Needs a trained critic at ./critic  (one-off:  python3 critic.py train).
"""
import json, math, sys
from pathlib import Path

import torch

from model.load import load_run
from critic import load_critic, pll, CRITIC_DIR

N_DEFAULT = 1000
max_new_tokens = 64        # titles early-stop at <eos> well before this
temperature = 1.0
top_k = 200
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    argv = sys.argv[1:]
    if not argv:
        sys.exit("usage: python3 eval_gen.py <run_dir> [N]")
    run_dir = argv[0]
    N = int(argv[1]) if len(argv) > 1 else N_DEFAULT

    torch.manual_seed(seed)
    model, tok, ckpt = load_run(run_dir, device=device)
    critic, craw, mask_id, pad_id = load_critic(CRITIC_DIR, device=device)
    block = critic.cfg.block_size

    eos_id = tok.tk.token_to_id("<eos>")
    if eos_id is None:
        eos_id = 1   # specials are [pad, eos, unk] -> eos is id 1 by construction

    # --- generate N titles, score each by critic pseudo-PLL, aggregate per character ---
    tot_nats, tot_chars, kept = 0.0, 0, 0
    for _ in range(N):
        x = torch.tensor([[eos_id]], dtype=torch.long, device=device)
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, eos_id=eos_id)
        text = tok.decode(y[0].tolist()).strip()
        if not text:
            continue
        cids = torch.tensor(craw.encode(text).ids[:block], dtype=torch.long, device=device)
        if cids.numel() == 0:
            continue
        tot_nats += pll(critic, cids, mask_id, pad_id)
        tot_chars += len(text)
        kept += 1

    nats_per_char = tot_nats / max(1, tot_chars)
    rec = {"run_dir": run_dir,
           "n_gen": kept,
           "critic_nats_per_char": round(nats_per_char, 4),
           "critic_pseudo_ppl": round(math.exp(nats_per_char), 4),
           "model_val_loss": ckpt.get("val_loss")}
    (Path(run_dir) / "eval.log").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
