//! CLI entry point for the from-scratch CPU inference engine.
//!
//! Wires the pieces together: parse args -> load the model + tokenizer -> run an
//! autoregressive generation loop -> stream decoded text to stdout. The actual maths
//! lives in `math`/`model`/`cache`/`sample`; this file is just orchestration.
//!
//! The engine reads the artifacts produced by `export/export.py`:
//!   <weights>/model.safetensors   the weights
//!   <weights>/meta.json           the model shape (ModelConfig)
//!   <weights>/tokenizer.json      the exact train-time tokenizer
//!
//!   cargo run --release -- generate --prompt "Show HN: "

#![allow(dead_code)] // modules are scaffolded ahead of being filled in

mod cache;
mod config;
mod math;
mod model;
mod sample;
mod tokenizer;

use std::error::Error;
use std::path::PathBuf;

use clap::{Parser, Subcommand};

use config::ModelConfig;
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

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Generate(args) => generate(&cli.weights, args),
    }
}

fn generate(weights: &PathBuf, args: GenArgs) -> Result<(), Box<dyn Error>> {
    let cfg = ModelConfig::load(weights.join("meta.json"))?;
    let tok = Tokenizer::load(weights.join("tokenizer.json"))?;
    eprintln!(
        "loaded config: {} layers, d_model {}, vocab {} (val_loss {:?})",
        cfg.n_layer, cfg.d_model, cfg.vocab_size, cfg.val_loss
    );

    let ids = tok.encode(&args.prompt);
    eprintln!("prompt -> {} tokens: {:?}", ids.len(), ids);

    // TODO(wiring): once model.rs is written, this becomes roughly:
    //   let model = Model::load(weights.join("model.safetensors"), &cfg)?;
    //   let mut cache = KvCache::new(&cfg);
    //   let mut rng = Rng::new(args.seed);
    //   let mut pos = 0;
    //   // prefill the prompt, then decode one token at a time
    //   for &id in &ids { model.forward(id, pos, &mut cache); pos += 1; }
    //   let mut last = *ids.last().unwrap();
    //   for _ in 0..args.max_tokens {
    //       let logits = model.forward(last, pos, &mut cache);
    //       last = sample::sample(&logits, args.temperature, args.top_k, &mut rng);
    //       print!("{}", tok.decode(&[last]));
    //       pos += 1;
    //       if last == cfg.eos_id { break; }
    //   }
    let _ = &args; // silence unused-field warnings until wired
    todo!("forward pass not implemented yet — see model.rs / sample.rs");
}
