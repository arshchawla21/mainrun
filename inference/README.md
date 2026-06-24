# inference

A tiny, from-scratch, CPU-only inference engine in Rust for the nanoGPT-style model
trained in this repo. No ML framework: the forward pass is hand-written f32, the only
real dependencies are the safetensors weight format and the HuggingFace tokenizer.

This is a side project — it is **not** part of the graded submission. It exists to play
with the trained model and to have something fun to write about.

## Prerequisites

A Rust toolchain (and a C linker, which most systems already have):

```bash
# user-local, no root — see https://rustup.rs
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

On Debian/Ubuntu, if `cc` is missing: `sudo apt-get install -y build-essential`.

## 1. Export a trained model

The Rust engine reads safetensors, not the PyTorch pickle. Convert a finished run once:

```bash
python export/export.py ../sweeps/<run_dir>      # writes ./weights/{model.safetensors,meta.json,tokenizer.json}
```

## 2. Build & run

Build once in release (the `--` separates cargo's args from the engine's):

```bash
cargo build --release
```

Then run the binary directly (or via `cargo run --release --`):

```bash
# basic generation
./target/release/inference generate --prompt "Show HN: "

# longer sample, custom sampling
./target/release/inference generate --prompt "Ask HN: " --max-tokens 120 --temperature 0.8 --top-k 40

# greedy / deterministic (argmax, ignores seed)
./target/release/inference generate --prompt "Show HN: " --temperature 0

# reproducible sampling: same seed -> identical output
./target/release/inference generate --prompt "The " --seed 42

# empty prompt: seeds with <eos> (the start token) and generates a fresh title
./target/release/inference generate --prompt ""

# point at a different export dir
./target/release/inference --weights weights generate --prompt "Show HN: "

# equivalently, through cargo
cargo run --release -- generate --prompt "Show HN: " --max-tokens 200
```

Flags: `--prompt` (text to continue), `--max-tokens` (default 200), `--temperature`
(default 0.8; `0` = greedy), `--top-k` (default 40; `0` = off), `--seed` (default 1337),
`--weights` (export dir, default `weights`). The prompt is echoed and new tokens stream to
stdout; progress/info lines go to stderr, so `2>/dev/null` gives just the generated text.

## Layout

| file            | what it does |
|-----------------|--------------|
| `src/main.rs`   | CLI; loads everything and runs the generation loop |
| `src/constants.rs` | compile-time model shape + a `meta.json` sanity check |
| `src/tokenizer.rs` | wrapper over the `tokenizers` crate (`tokenizer.json`) |
| `src/math.rs`   | numeric kernels: matvec, rmsnorm, gelu, softmax, rope |
| `src/model.rs`  | weight loading + the forward pass |
| `src/cache.rs`  | the KV cache |
| `src/sample.rs` | logits -> next token (temperature, top-k) |
| `export/export.py` | `model.pt` -> safetensors + meta.json |

# Acknowledgements
https://github.com/srush/llama2.rs