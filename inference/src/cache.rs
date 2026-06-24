//! The KV cache: per-layer rolling buffers of the rope'd keys and values, so each new
//! token attends over all prior positions without recomputing them. Sized to SEQ_LEN;
//! context is capped there for v1 (the model's trained window).
//!
//! Layout is flat: `keys[layer]` grows by KV_DIM f32 per stored position, so position `t`'s
//! key for head `h` is `keys[layer][t*KV_DIM + h*HEAD_SIZE ..][..HEAD_SIZE]`. Full MHA here,
//! so KV_DIM == DIM and there are N_HEADS distinct K/V heads.

use crate::constants::{KV_DIM, N_LAYERS, SEQ_LEN};

pub struct KvCache {
    keys: Vec<Vec<f32>>, // [N_LAYERS][pos*KV_DIM ..]
    vals: Vec<Vec<f32>>, // [N_LAYERS][pos*KV_DIM ..]
}

impl KvCache {
    pub fn new() -> Self {
        let mk = || (0..N_LAYERS).map(|_| Vec::with_capacity(SEQ_LEN * KV_DIM)).collect();
        Self { keys: mk(), vals: mk() }
    }

    /// Append the current token's rope'd key/value (each KV_DIM long) for `layer`.
    pub fn push(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        self.keys[layer].extend_from_slice(k);
        self.vals[layer].extend_from_slice(v);
    }

    pub fn keys(&self, layer: usize) -> &[f32] {
        &self.keys[layer]
    }

    pub fn vals(&self, layer: usize) -> &[f32] {
        &self.vals[layer]
    }
}
