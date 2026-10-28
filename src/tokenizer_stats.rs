/// Statistics about a tokenizer's vocabulary
#[derive(Debug)]
pub struct BytePairEncodingTokenizerStats {
    /// Total vocabulary size (base tokens + learned merges)
    pub vocab_size: usize,
    /// Number of merge rules learned
    pub num_merges: usize,
    /// Number of base tokens (always 256 for byte-level BPE)
    pub base_tokens: usize,
}
