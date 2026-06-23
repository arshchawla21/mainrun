//! The KV cache: per-layer rolling buffers of the rope'd keys and values, so each new
//! token attends over all prior positions without recomputing them. Sized to block_size;
//! context is capped there for v1 (the model's trained window).
