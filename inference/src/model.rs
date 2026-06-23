//! The model: weight storage, loading from safetensors, and the forward pass.
//!
//! `Weights` mmaps model.safetensors and views each tensor as &[f32]. `forward` runs one
//! token through all layers (gpt_v2 Block: rmsnorm -> output-gated attention -> layerscale
//! residual -> rmsnorm -> GELU MLP -> layerscale residual), reads/writes the KV cache, and
//! returns logits via the tied embedding head. Single sequence => attention is plain causal.
