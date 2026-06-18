"""
Autoregressive decoder

Run with: `python3 sample.py <dir> [n] [t] [k]`

dir (str): run directory; must contain model.pt + tokenizer.json (as written by save_run).
n (int):   number of samples to generate
t (float): sampling temperature  (1.0 = unchanged, <1 = less random, >1 = more random)
k (int):   top-k  (retain only the k most likely tokens each step)

Unconditional generation: titles are trained as  title1<eos>title2<eos>...  so <eos> always
precedes a fresh title -> it is the de-facto start token. We prime with a single <eos> and stop
each sample at the next <eos> (the title boundary).

Adapted from https://github.com/karpathy/nanoGPT/blob/master/sample.py
"""

import sys

import torch

from model.load import load_run

# --- defaults (overridable via CLI args) ---
num_samples = 10        # number of samples to draw
max_new_tokens = 500    # hard cap per sample (early-stops at <eos> well before this for titles)
temperature = 0.8       # 1.0 = no change, < 1.0 = less random, > 1.0 = more random
top_k = 200             # retain only the top_k most likely tokens, others clamped to 0 probability
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    # --- parse: sample.py <dir> [n] [t] [k] ---
    argv = sys.argv[1:]
    if not argv:
        sys.exit("usage: python3 sample.py <run_dir> [n_samples] [temperature] [top_k]")
    run_dir = argv[0]
    n = int(argv[1])   if len(argv) > 1 else num_samples
    t = float(argv[2]) if len(argv) > 2 else temperature
    k = int(argv[3])   if len(argv) > 3 else top_k

    torch.manual_seed(seed)

    # --- load model.pt + tokenizer.json from dir (picks up the exact tokenizer used at train) ---
    model, tok, ckpt = load_run(run_dir, device=device)
    print(f"loaded {run_dir}\n  val_loss={ckpt.get('val_loss'):.4f}  vocab={tok.vocab_size}  "
          f"block_size={model.cfg.block_size}  device={device}\n")

    eos_id = tok.tk.token_to_id("<eos>")
    if eos_id is None:
        eos_id = 1   # specials are [pad, eos, unk] -> eos is id 1 by construction

    # --- generate, conditioned on the start-of-title <eos>, one title per sample ---
    for i in range(n):
        x = torch.tensor([[eos_id]], dtype=torch.long, device=device)
        y = model.generate(x, max_new_tokens, temperature=t, top_k=k, eos_id=eos_id)
        text = tok.decode(y[0].tolist()).strip()   # skip_special_tokens drops the priming/closing <eos>
        print(f"[{i + 1:>2}] {text} |")
        print('---------------')


if __name__ == "__main__":
    main()
