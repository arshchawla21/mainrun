"""
Autoregressive decoder

Run with: `python3 sample.py <dir> [-n N] [-t TEMP] [-k TOPK] [-p PROMPT]`

dir (str):       run directory; must contain model.pt + tokenizer.json (as written by save_run).
-n/--num (int):  number of samples to generate
-t/--temp (float): sampling temperature  (1.0 = unchanged, <1 = less random, >1 = more random)
-k/--topk (int): top-k  (retain only the k most likely tokens each step)
-p/--prompt (str): optional context to condition on (e.g. "Show HN: "); omit for unconditional.

Titles are trained as  title1<eos>title2<eos>...  so <eos> always precedes a fresh title -> it is
the de-facto start token. We prime with a single <eos>, then append the prompt tokens (if any), and
let the model continue; each sample stops at the next <eos> (the title boundary).

Adapted from https://github.com/karpathy/nanoGPT/blob/master/sample.py
"""

import argparse

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
    ap = argparse.ArgumentParser(description="autoregressive sampler")
    ap.add_argument("run_dir", help="run directory (model.pt + tokenizer.json)")
    ap.add_argument("-n", "--num", type=int, default=num_samples, help="number of samples")
    ap.add_argument("-t", "--temp", type=float, default=temperature, help="temperature")
    ap.add_argument("-k", "--topk", type=int, default=top_k, help="top-k")
    ap.add_argument("-p", "--prompt", type=str, default="", help="context to condition on")
    ap.add_argument("--seed", type=int, default=seed)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # --- load model.pt + tokenizer.json from dir (picks up the exact tokenizer used at train) ---
    model, tok, ckpt = load_run(args.run_dir, device=device)
    print(f"loaded {args.run_dir}\n  val_loss={ckpt.get('val_loss'):.4f}  vocab={tok.vocab_size}  "
          f"block_size={model.cfg.block_size}  device={device}")
    if args.prompt:
        print(f"  prompt={args.prompt!r}")
    print()

    eos_id = tok.tk.token_to_id("<eos>")
    if eos_id is None:
        eos_id = 1   # specials are [pad, eos, unk] -> eos is id 1 by construction

    # prime with <eos> (start-of-title), then the prompt tokens (if any) -> the model continues it
    prime = [eos_id] + tok.encode(args.prompt)

    # --- generate, one title per sample ---
    for i in range(args.num):
        x = torch.tensor([prime], dtype=torch.long, device=device)
        y = model.generate(x, max_new_tokens, temperature=args.temp, top_k=args.topk, eos_id=eos_id)
        text = tok.decode(y[0].tolist()).strip()   # skip_special_tokens drops the priming/closing <eos>
        print(f"[{i + 1:>2}] {text} |")
        print('---------------')


if __name__ == "__main__":
    main()
