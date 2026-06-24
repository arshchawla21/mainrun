//! The numeric kernels: matvec (y = W·x [+b]), rmsnorm, exact-erf gelu, sigmoid,
//! softmax, dot, and the RoPE rotation. Plain f32, no SIMD yet -- correctness first.
//! Everything operates on flat &[f32] slices so the model code stays allocation-light.

use std::f32::consts::FRAC_1_SQRT_2;

use rayon::prelude::*;

use crate::constants::{HEAD_SIZE, ROPE_BASE};

/// Below this many output rows, the rayon fan-out costs more than it saves (e.g. the
/// 8-wide output gate), so we stay serial.
const PAR_THRESHOLD: usize = 64;

/// y = W·x (+ bias). W is row-major (out_features, in_features), so row `o` is
/// `w[o*n_in .. o*n_in + n_in]`. `n_in` is taken from `x`, `n_out` from `y`. Each output
/// row is an independent dot product, so we fan them across cores with rayon.
pub fn matvec(y: &mut [f32], w: &[f32], x: &[f32], bias: Option<&[f32]>) {
    let n_in = x.len();
    debug_assert_eq!(w.len(), y.len() * n_in);
    let row = |o: usize, y_o: &mut f32| {
        *y_o = bias.map_or(0.0, |b| b[o]) + dot(&w[o * n_in..o * n_in + n_in], x);
    };
    if y.len() >= PAR_THRESHOLD {
        y.par_iter_mut().enumerate().for_each(|(o, y_o)| row(o, y_o));
    } else {
        y.iter_mut().enumerate().for_each(|(o, y_o)| row(o, y_o));
    }
}

/// RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight  (no mean-subtraction, no bias).
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let ss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    for ((o, &x_i), &w_i) in out.iter_mut().zip(x).zip(weight) {
        *o = w_i * scale * x_i;
    }
}

/// Exact GELU (matches torch's default nn.GELU()): 0.5·x·(1 + erf(x/√2)).
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * FRAC_1_SQRT_2))
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// In-place softmax over the slice (max-shifted for numerical stability).
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// NeoX-style RoPE applied in place to a packed [n_heads · HEAD_SIZE] vector.
///
/// For each head, element j (in the first half) pairs with j + HEAD_SIZE/2, both rotated by
/// angle `pos · base^(-2j/HEAD_SIZE)`. Rotating the pair together avoids the in-place aliasing
/// that a single 0..HEAD_SIZE sweep would hit. Equivalent to gpt.py's cat(freqs,freqs)+rotate_half.
pub fn rope(vec: &mut [f32], pos: usize, n_heads: usize) {
    let half = HEAD_SIZE / 2;
    for h in 0..n_heads {
        let base = h * HEAD_SIZE;
        for j in 0..half {
            let inv_freq = ROPE_BASE.powf(-(2.0 * j as f32) / HEAD_SIZE as f32);
            let (sin, cos) = (pos as f32 * inv_freq).sin_cos();
            let a = vec[base + j];
            let b = vec[base + half + j];
            vec[base + j] = a * cos - b * sin;
            vec[base + half + j] = b * cos + a * sin;
        }
    }
}
