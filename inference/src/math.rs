//! The numeric kernels: matvec (y = W·x [+b]), rmsnorm, exact-erf gelu, sigmoid,
//! softmax, and the RoPE rotation. Plain f32, no SIMD yet — correctness first.
//! Everything operates on flat &[f32] slices so the model code stays allocation-light.
