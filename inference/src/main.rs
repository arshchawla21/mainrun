//! CLI entry point for the from-scratch CPU inference engine.
//!
//! Wires the pieces together: parse args -> load the model + tokenizer -> run an
//! autoregressive generation loop -> stream decoded text to stdout. The actual maths
//! lives in `math`/`model`/`cache`/`sample`; this file is just orchestration.
//!
//! The model shape is compile-time (`constants.rs`); the engine reads two artifacts
//! produced by `export/export.py`:
//!   <weights>/model.safetensors   the weights
//!   <weights>/tokenizer.json      the exact train-time tokenizer
//! plus <weights>/meta.json, used only to sanity-check the compiled-in dims.
//!
//!   cargo run --release -- generate --prompt "Show HN: "

#![allow(dead_code)] // modules are scaffolded ahead of being filled in

mod cache;
mod constants;
mod math;
mod model;
mod sample;
mod tokenizer;

use std::error::Error;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};

use cache::KvCache;
use model::Model;
use sample::{Rng, sample};
use tokenizer::Tokenizer;

#[derive(Parser)]
#[command(name = "inference", about = "A tiny from-scratch CPU LLM inference engine in Rust")]
struct Cli {
    /// Directory holding model.safetensors, meta.json and tokenizer.json
    #[arg(long, default_value = "weights", global = true)]
    weights: PathBuf,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate text from a prompt
    Generate(GenArgs),
    /// Benchmark steady-state decode throughput (loads once, warms up, times N tokens).
    /// Prints one JSON line on stdout; info goes to stderr.
    Bench(BenchArgs),
}

#[derive(Parser)]
struct GenArgs {
    /// The prompt to continue
    #[arg(long, default_value = "")]
    prompt: String,

    /// Maximum number of new tokens to sample
    #[arg(long, default_value_t = 200)]
    max_tokens: usize,

    /// Sampling temperature (0.0 = greedy / argmax)
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    /// Top-k filtering (0 = disabled)
    #[arg(long, default_value_t = 40)]
    top_k: usize,

    /// RNG seed for reproducible sampling
    #[arg(long, default_value_t = 1337)]
    seed: u64,
}

#[derive(Parser)]
struct BenchArgs {
    /// Prompt to prefill before timing
    #[arg(long, default_value = "Show HN: ")]
    prompt: String,

    /// Number of tokens to time (greedy decode)
    #[arg(long, default_value_t = 128)]
    tokens: usize,

    /// Untimed warmup tokens before measurement
    #[arg(long, default_value_t = 16)]
    warmup: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Generate(args) => generate(&cli.weights, args),
        Cmd::Bench(args) => bench(&cli.weights, args),
    }
}

fn generate(weights: &PathBuf, args: GenArgs) -> Result<(), Box<dyn Error>> {
    // confirm the weights on disk match the model this binary was compiled for
    constants::assert_matches_meta(weights.join("meta.json"))?;
    let tok = Tokenizer::load(weights.join("tokenizer.json"))?;
    let model = Model::load(weights.join("model.safetensors"))?;
    let eos = tok.eos_id();
    eprintln!(
        "model: {} layers, dim {}, vocab {}",
        constants::N_LAYERS,
        constants::DIM,
        constants::VOCAB_SIZE
    );

    let ids = tok.encode(&args.prompt);
    eprintln!("prompt -> {} tokens", ids.len());

    let mut cache = KvCache::new();
    let mut rng = Rng::new(args.seed);
    let mut pos = 0usize;
    let mut logits = Vec::new();

    // prefill: run the prompt through, keeping the logits after the last token. An empty
    // prompt is seeded with <eos>, the de-facto start token (matching sample.py).
    print!("{}", args.prompt);
    if ids.is_empty() {
        logits = model.forward(eos, pos, &mut cache);
        pos += 1;
    } else {
        for &id in &ids {
            logits = model.forward(id, pos, &mut cache);
            pos += 1;
        }
    }

    // decode one token at a time, streaming the text out
    for _ in 0..args.max_tokens {
        if pos >= constants::SEQ_LEN {
            break;
        }
        let next = sample(&logits, args.temperature, args.top_k, &mut rng);
        if next == eos {
            break;
        }
        print!("{}", tok.decode(&[next]));
        std::io::stdout().flush().ok();
        logits = model.forward(next, pos, &mut cache);
        pos += 1;
    }
    println!();
    Ok(())
}

fn bench(weights: &PathBuf, args: BenchArgs) -> Result<(), Box<dyn Error>> {
    constants::assert_matches_meta(weights.join("meta.json"))?;
    let tok = Tokenizer::load(weights.join("tokenizer.json"))?;
    let load_start = Instant::now();
    let model = Model::load(weights.join("model.safetensors"))?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    let eos = tok.eos_id();

    let ids = tok.encode(&args.prompt);
    let mut cache = KvCache::new();
    let mut rng = Rng::new(0); // unused: greedy decode below
    let mut pos = 0usize;

    // prefill
    let mut logits = model.forward(if ids.is_empty() { eos } else { ids[0] }, pos, &mut cache);
    pos += 1;
    for &id in ids.iter().skip(1) {
        logits = model.forward(id, pos, &mut cache);
        pos += 1;
    }

    // greedy step: argmax via sample() with temperature 0
    let mut step = |logits: &[f32], pos: usize, cache: &mut KvCache| -> Vec<f32> {
        let next = sample(logits, 0.0, 0, &mut rng);
        model.forward(next, pos, cache)
    };

    for _ in 0..args.warmup {
        if pos >= constants::SEQ_LEN {
            break;
        }
        logits = step(&logits, pos, &mut cache);
        pos += 1;
    }

    let timed_start = Instant::now();
    let mut n = 0usize;
    for _ in 0..args.tokens {
        if pos >= constants::SEQ_LEN {
            break;
        }
        logits = step(&logits, pos, &mut cache);
        pos += 1;
        n += 1;
    }
    let total_s = timed_start.elapsed().as_secs_f64();

    eprintln!(
        "rust: loaded in {load_ms:.0} ms, decoded {n} tokens in {total_s:.3} s",
    );
    println!(
        "{{\"engine\":\"rust\",\"tokens\":{n},\"load_ms\":{load_ms:.1},\"total_s\":{total_s:.4},\"tok_per_s\":{:.2},\"ms_per_tok\":{:.3}}}",
        n as f64 / total_s,
        total_s * 1000.0 / n as f64,
    );
    Ok(())
}
