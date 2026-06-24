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
        // tokenizers returns Box<dyn Error + Send + Sync>, which `?` can't coerce into
        // our Box<dyn Error>; flatten it to a string error.
        let inner = HfTokenizer::from_file(path).map_err(|e| e.to_string())?;
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

    /// Id of the `<eos>` token (the title delimiter / generation stop token).
    pub fn eos_id(&self) -> u32 {
        self.inner
            .token_to_id("<eos>")
            .expect("tokenizer has no <eos> token")
    }
}
