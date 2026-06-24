//! Turning logits into the next token id: temperature scaling, optional top-k, then a
//! multinomial draw (temperature 0 => greedy argmax). Includes a small seeded PRNG so runs
//! are reproducible without pulling in the `rand` crate.

use crate::math::softmax;

/// SplitMix64 -> uniform f32 in [0, 1). Tiny, fast, deterministic from a seed.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) from the top 24 bits.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// Pick the next token from `logits`. `temperature <= 0` => greedy argmax. Otherwise scale
/// by temperature, optionally keep only the top-k logits (`top_k == 0` disables), softmax,
/// and draw multinomially. Matches sample.py: top-k keeps logits >= the k-th largest.
pub fn sample(logits: &[f32], temperature: f32, top_k: usize, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits);
    }

    let mut probs: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();

    if top_k > 0 && top_k < probs.len() {
        let mut sorted = probs.clone();
        // descending order; element at index top_k-1 is the k-th largest -> the keep threshold
        sorted.select_nth_unstable_by(top_k - 1, |a, b| b.partial_cmp(a).unwrap());
        let kth = sorted[top_k - 1];
        for p in probs.iter_mut() {
            if *p < kth {
                *p = f32::NEG_INFINITY;
            }
        }
    }

    softmax(&mut probs);

    // inverse-CDF draw
    let r = rng.next_f32();
    let mut cdf = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cdf += p;
        if r < cdf {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32 // fallthrough (rounding)
}

fn argmax(x: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best as u32
}
