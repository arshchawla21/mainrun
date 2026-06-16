"""
Analyse the distribution of the NL data,
to inform tokeniser design.

changelog:
  import key functions from train (16/06)
"""
from collections import Counter

from train import get_titles, Hyperparameters
from model.tokenizer import train_tokenizer

from pathlib import Path

import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

VOCAB_SIZES = [1000, 2000, 4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000]
TERMS = ["JavaScript", "PostgreSQL", "Kubernetes",
        "open-source", "LLM", "API", "GitHub",
        "machine learning", "Show HN", "Ask HN"]

def evaluate_tokenizer(tokenizer, corpus, low_freq_threshold=5):
    encodings = [tokenizer.encode(t).ids for t in corpus]
    lengths   = [len(e) for e in encodings]
    n_chars   = sum(len(t) for t in corpus)
    n_tokens  = sum(lengths)
 
    # token frequency across the whole corpus
    freq       = Counter(tok for e in encodings for tok in e)
    vocab_size = tokenizer.get_vocab_size()
    used       = len(freq)
 
    # split the two failure modes apart so they don't overlap:
    rare   = sum(1 for v in freq.values() if v < low_freq_threshold)  # appears 1..threshold-1
    unused = vocab_size - used                                        # appears 0 times
 
    return {
        "vocab_size":        vocab_size,
        "fertility":         n_tokens / n_chars,            # tokens per char
        "mean_seq_len":      float(np.mean(lengths)),
        "p95_seq_len":       float(np.percentile(lengths, 95)),
        "vocab_utilisation": used / vocab_size,             # want close to 1.0
        "rare_token_pct":    rare / vocab_size,             # want close to 0
        "unused_token_pct":  unused / vocab_size,           # want close to 0
    }

def plot_results(vocab_sizes, results, path="../results/vocab_analysis.png"):
    fertility = [r["fertility"] for r in results]
    rare      = [r["rare_token_pct"] for r in results]
 
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(vocab_sizes, fertility, "o-",  color="#378ADD",
            label="fertility (tokens/char, lower=better)")
    ax.plot(vocab_sizes, rare, "s--", color="#D85A30",
            label="rare token % (lower=better)")
 
    ax.set_xscale("log", base=2)
    ax.set_xticks(vocab_sizes)
    ax.xaxis.set_major_formatter(ScalarFormatter())  # plain numbers, not 2^n
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlabel("vocab size")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 0.75)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
 
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"saved plot to {path}")


def tokenization_report(tokenizer, terms):
    report = {}
    for term in terms:
        toks = [tokenizer.id_to_token(i) for i in tokenizer.encode(term).ids]
        report[term] = toks
        print(f"{term:25s} -> {toks}")
    return report

def save_results(results, path="../results/vocab_qualitative_analysis.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved results to {path}")

def main():
    args = Hyperparameters()
    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)

    for name, titles in [("train", train_titles), ("val", val_titles)]:
        print(f"len({name})={len(titles)}")


    char_lengths = [len(x) for x in train_titles]
    print(f"average char length (train)={np.mean(char_lengths):.1f}")

    results = []

    for vc in VOCAB_SIZES:
        # train on train only, so no leakage in the selection metrics
        tok = train_tokenizer(train_titles, vc, eos_token="<eos>")

        # analyse metrics
        stats = evaluate_tokenizer(tok, val_titles)
        print(
            f"vocab={vc:6d} | fertility={stats['fertility']:.3f} | "
            f"mean_len={stats['mean_seq_len']:.1f} | "
            f"util={stats['vocab_utilisation']:.2f} | "
            f"rare={stats['rare_token_pct']:.2f} | "
            f"unused={stats['unused_token_pct']:.2f}"
        )

        # qualitative analysis
        stats['tokenization'] = tokenization_report(tok, TERMS)
        results.append(stats)

    save_results(results)
    plot_results(VOCAB_SIZES, results)


if __name__ == '__main__':
    main()