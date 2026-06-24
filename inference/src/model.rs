//! The model: weight storage, loading from safetensors, and the forward pass.
//!
//! `Model` loads model.safetensors into owned f32 buffers. `forward` runs one token through
//! all layers -- the gpt_v2 Block with the Phase B winners hardwired:
//!   rmsnorm -> output-gated MHA -> layerscale residual -> rmsnorm -> GELU MLP -> layerscale
//! -- reads/writes the KV cache, and returns logits via the tied embedding head. A single
//! sequence is decoded, so attention is plain causal (title-masking is irrelevant here).

use std::error::Error;
use std::path::Path;

use memmap2::Mmap;
use safetensors::SafeTensors;

use crate::cache::KvCache;
use crate::constants::*;
use crate::math::{dot, gelu, matvec, rmsnorm, rope, sigmoid, softmax};

/// All weights for one transformer block (shapes in comments are torch's [out, in]).
struct Layer {
    ln1: Vec<f32>,        // [DIM]            attention RMSNorm
    ln2: Vec<f32>,        // [DIM]            MLP RMSNorm
    qkv_w: Vec<f32>,      // [3*DIM, DIM]
    qkv_b: Vec<f32>,      // [3*DIM]
    proj_w: Vec<f32>,     // [DIM, DIM]
    proj_b: Vec<f32>,     // [DIM]
    gate_w: Vec<f32>,     // [N_HEADS, DIM]   output gate (per-head logit)
    gate_b: Vec<f32>,     // [N_HEADS]
    mlp_fc_w: Vec<f32>,   // [MLP_HIDDEN, DIM]
    mlp_fc_b: Vec<f32>,   // [MLP_HIDDEN]
    mlp_proj_w: Vec<f32>, // [DIM, MLP_HIDDEN]
    mlp_proj_b: Vec<f32>, // [DIM]
    ls1: Vec<f32>,        // [DIM]            layerscale on the attention residual
    ls2: Vec<f32>,        // [DIM]            layerscale on the MLP residual
}

pub struct Model {
    token_emb: Vec<f32>, // [VOCAB_SIZE, DIM]; also the tied output head
    layers: Vec<Layer>,
    ln_f: Vec<f32>,      // [DIM]             final RMSNorm
}

impl Model {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let st = SafeTensors::deserialize(&mmap)?;

        // pull a tensor by name into an owned Vec<f32> (export writes everything as f32 LE)
        let get = |name: &str| -> Vec<f32> {
            let t = st
                .tensor(name)
                .unwrap_or_else(|_| panic!("missing tensor {name} in safetensors"));
            t.data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };

        let layers = (0..N_LAYERS)
            .map(|i| {
                let p = |s: &str| get(&format!("blocks.{i}.{s}"));
                Layer {
                    ln1: p("ln1.weight"),
                    ln2: p("ln2.weight"),
                    qkv_w: p("attn.qkv.weight"),
                    qkv_b: p("attn.qkv.bias"),
                    proj_w: p("attn.proj.weight"),
                    proj_b: p("attn.proj.bias"),
                    gate_w: p("attn.gate.weight"),
                    gate_b: p("attn.gate.bias"),
                    mlp_fc_w: p("mlp.net.0.weight"),
                    mlp_fc_b: p("mlp.net.0.bias"),
                    mlp_proj_w: p("mlp.net.2.weight"),
                    mlp_proj_b: p("mlp.net.2.bias"),
                    ls1: p("ls1"),
                    ls2: p("ls2"),
                }
            })
            .collect();

        Ok(Self {
            token_emb: get("token_emb.weight"),
            layers,
            ln_f: get("ln_f.weight"),
        })
    }

    /// One decode step: embed `token` at position `pos`, run all layers (updating `cache`),
    /// and return the [VOCAB_SIZE] logits. `pos` must be < SEQ_LEN.
    pub fn forward(&self, token: u32, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        let scale = 1.0 / (HEAD_SIZE as f32).sqrt();
        // residual stream, seeded with the token embedding
        let mut x = self.token_emb[token as usize * DIM..][..DIM].to_vec();

        for (l, layer) in self.layers.iter().enumerate() {
            // --- attention ---
            let mut xn = vec![0.0f32; DIM];
            rmsnorm(&mut xn, &x, &layer.ln1, RMS_EPS);

            // qkv projection, then split into q | k | v (each DIM long)
            let mut qkv = vec![0.0f32; 3 * DIM];
            matvec(&mut qkv, &layer.qkv_w, &xn, Some(&layer.qkv_b));
            rope(&mut qkv[0..DIM], pos, N_HEADS);
            rope(&mut qkv[DIM..2 * DIM], pos, N_HEADS);
            cache.push(l, &qkv[DIM..2 * DIM], &qkv[2 * DIM..3 * DIM]);

            // causal attention per head, over all positions 0..=pos in the cache
            let q = &qkv[0..DIM];
            let keys = cache.keys(l);
            let vals = cache.vals(l);
            let mut attn = vec![0.0f32; DIM];
            for h in 0..N_HEADS {
                let qh = &q[h * HEAD_SIZE..][..HEAD_SIZE];
                let mut scores = vec![0.0f32; pos + 1];
                for (t, s) in scores.iter_mut().enumerate() {
                    let kh = &keys[t * KV_DIM + h * HEAD_SIZE..][..HEAD_SIZE];
                    *s = dot(qh, kh) * scale;
                }
                softmax(&mut scores);
                let oh = &mut attn[h * HEAD_SIZE..][..HEAD_SIZE];
                for (t, &s) in scores.iter().enumerate() {
                    let vh = &vals[t * KV_DIM + h * HEAD_SIZE..][..HEAD_SIZE];
                    for (o, &v_i) in oh.iter_mut().zip(vh) {
                        *o += s * v_i;
                    }
                }
            }

            // output gate: per-head sigmoid(gate·xn) scales that head's output
            let mut gate = vec![0.0f32; N_HEADS];
            matvec(&mut gate, &layer.gate_w, &xn, Some(&layer.gate_b));
            for h in 0..N_HEADS {
                let g = sigmoid(gate[h]);
                for o in &mut attn[h * HEAD_SIZE..][..HEAD_SIZE] {
                    *o *= g;
                }
            }

            // output projection, then layerscale residual
            let mut attn_out = vec![0.0f32; DIM];
            matvec(&mut attn_out, &layer.proj_w, &attn, Some(&layer.proj_b));
            for ((x_i, &ls), &a) in x.iter_mut().zip(&layer.ls1).zip(&attn_out) {
                *x_i += ls * a;
            }

            // --- GELU MLP ---
            let mut xn2 = vec![0.0f32; DIM];
            rmsnorm(&mut xn2, &x, &layer.ln2, RMS_EPS);
            let mut hidden = vec![0.0f32; MLP_HIDDEN];
            matvec(&mut hidden, &layer.mlp_fc_w, &xn2, Some(&layer.mlp_fc_b));
            for v in &mut hidden {
                *v = gelu(*v);
            }
            let mut mlp_out = vec![0.0f32; DIM];
            matvec(&mut mlp_out, &layer.mlp_proj_w, &hidden, Some(&layer.mlp_proj_b));
            for ((x_i, &ls), &m) in x.iter_mut().zip(&layer.ls2).zip(&mlp_out) {
                *x_i += ls * m;
            }
        }

        // final norm + tied head (token_emb doubles as the output projection, no bias)
        let mut xf = vec![0.0f32; DIM];
        rmsnorm(&mut xf, &x, &self.ln_f, RMS_EPS);
        let mut logits = vec![0.0f32; VOCAB_SIZE];
        matvec(&mut logits, &self.token_emb, &xf, None);
        logits
    }
}
