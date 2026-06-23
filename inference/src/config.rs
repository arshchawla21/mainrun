//! The model shape, deserialized from `meta.json` (written by `export/export.py`).
//!
//! `meta.json` is just the run's GPTConfig kwargs plus `val_loss`, so the Rust engine
//! is data-driven: change the exported run and the dims follow, no recompile. Unknown
//! fields in the JSON are ignored, so training-only knobs (dropout, lr, ...) are harmless.
//! Architectural constants that never vary (RoPE base, RMSNorm eps) live here as consts.

use std::error::Error;
use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub d_model: usize,

    /// Phase B attention variant: "mha" | "output_gated" | "value_residual" | ...
    #[serde(default = "default_attn")]
    pub attn_type: String,
    /// Phase B cross-layer path: "none" | "layerscale" | "unet" | "embedding_shortcut"
    #[serde(default = "default_residual")]
    pub residual: String,

    /// attention confined within each <eos>-delimited title (irrelevant for single-prompt decode)
    #[serde(default)]
    pub title_masking: bool,
    /// token id of <eos>; also the stop token for generation
    #[serde(default)]
    pub eos_id: u32,

    /// carried through for logging; not used by the forward pass
    #[serde(default)]
    pub val_loss: Option<f32>,
}

fn default_attn() -> String {
    "mha".to_string()
}
fn default_residual() -> String {
    "none".to_string()
}

impl ModelConfig {
    /// RoPE frequency base (RotaryEmbedding default).
    pub const ROPE_BASE: f32 = 10_000.0;
    /// RMSNorm epsilon.
    pub const RMS_EPS: f32 = 1e-6;

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_head
    }

    /// GELU MLP hidden width (4 * d_model).
    pub fn mlp_hidden(&self) -> usize {
        4 * self.d_model
    }
}
