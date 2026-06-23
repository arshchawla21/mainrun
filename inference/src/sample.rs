//! Turning logits into the next token id: temperature scaling, optional top-k, then a
//! multinomial draw (temperature 0 => greedy argmax). Includes a small seeded PRNG so runs
//! are reproducible without pulling in the `rand` crate.
