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

```bash
cargo run --release -- generate --prompt "Show HN: " --max-tokens 200
```

## Layout

| file            | what it does |
|-----------------|--------------|
| `src/main.rs`   | CLI; loads everything and runs the generation loop |
| `src/config.rs` | `ModelConfig`, deserialized from `meta.json` |
| `src/tokenizer.rs` | wrapper over the `tokenizers` crate (`tokenizer.json`) |
| `src/math.rs`   | numeric kernels: matvec, rmsnorm, gelu, softmax, rope |
| `src/model.rs`  | weight loading + the forward pass |
| `src/cache.rs`  | the KV cache |
| `src/sample.rs` | logits → next token (temperature, top-k) |
| `export/export.py` | `model.pt` → safetensors + meta.json |
