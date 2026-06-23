// --- swept dims (must match the exported run) ---
pub const DIM: usize = 512; // d_model
pub const N_LAYERS: usize = 12;
pub const N_HEADS: usize = 8;
pub const VOCAB_SIZE: usize = 16_000;
pub const SEQ_LEN: usize = 256; // block_size; also the context/KV-cache cap for v1

// --- derived dims ---
pub const HEAD_SIZE: usize = DIM / N_HEADS; // 64
pub const MLP_HIDDEN: usize = 4 * DIM; // GELU MLP inner width

// --- MHA ---
pub const N_KV_HEADS: usize = N_HEADS;
pub const KV_DIM: usize = DIM;

// --- architectural constants (never swept) ---
pub const ROPE_BASE: f32 = 10_000.0; // RotaryEmbedding default
pub const RMS_EPS: f32 = 1e-6; // RMSNorm epsilon

/// Sanity-check the compiled-in dims against the `meta.json` exported with the weights.
/// Cheap insurance against running an engine built for a different model than the one on disk.
pub fn assert_matches_meta(path: impl AsRef<std::path::Path>) -> Result<(), Box<dyn std::error::Error>> {
    #[derive(serde::Deserialize)]
    struct Meta {
        vocab_size: usize,
        block_size: usize,
        n_layer: usize,
        n_head: usize,
        d_model: usize,
    }
    let m: Meta = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    assert_eq!(m.d_model, DIM, "d_model mismatch: rebuild constants.rs for this model");
    assert_eq!(m.n_layer, N_LAYERS, "n_layer mismatch: rebuild constants.rs for this model");
    assert_eq!(m.n_head, N_HEADS, "n_head mismatch: rebuild constants.rs for this model");
    assert_eq!(m.vocab_size, VOCAB_SIZE, "vocab_size mismatch: rebuild constants.rs for this model");
    assert_eq!(m.block_size, SEQ_LEN, "block_size mismatch: rebuild constants.rs for this model");
    Ok(())
}
