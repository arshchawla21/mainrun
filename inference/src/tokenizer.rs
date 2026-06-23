//! Thin wrapper over the HuggingFace `tokenizers` crate.
//!
//! Loads the exact `tokenizer.json` saved at train time (the winner uses a unigram
//! tokenizer with a ByteLevel pre-tokenizer/decoder), so encode/decode match training
//! byte-for-byte. We add no logic of our own — just a small surface the engine calls.

use std::error::Error;
use std::path::Path;

use tokenizers::Tokenizer as HfTokenizer;

pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let inner = HfTokenizer::from_file(path)?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.inner
            .encode(text, false)
            .expect("tokenizer encode failed")
            .get_ids()
            .to_vec()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner
            .decode(ids, true)
            .expect("tokenizer decode failed")
    }
}
