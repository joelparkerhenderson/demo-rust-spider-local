//! Byte Pair Encoding (BPE) Tokenizer
//!
//! <https://github.com/tag1consulting/feste/blob/main/src/tokenizer.rs>
//!
//! This module implements BPE tokenization from scratch. BPE is the standard
//! tokenization method used by GPT-2, GPT-3, and many other language models.
//!
//! ## How BPE Works
//!
//! 1. **Start with byte-level encoding**: 256 base tokens (one per byte value: 0-255)
//! 2. **Count adjacent pairs**: Find the most common adjacent pair in the corpus
//! 3. **Merge the most frequent pair**: Create a new token representing that pair
//! 4. **Repeat**: Continue until vocabulary reaches the target size
//!
//! ## Example
//!
//! Given corpus: "hello hello world"
//! - Iteration 1: Most common pair = ('l','l') → merge to token 256: "he\[256\]o he\[256\]o world"
//! - Iteration 2: Most common pair = ('h','e') → merge to token 257: "\[257\]\[256\]o \[257\]\[256\]o world"
//! - Iteration 3: Most common pair = (257, 256) → merge to token 258: "\[258\]o \[258\]o world"
//! - Continue until vocab_size is reached...
//!
//! ## How Tokens Are Represented
//!
//! A token is an **integer ID** ([`TokenId`](crate::tokenizer_functions::TokenId), a `u32`), not a string.
//!
//! - IDs `0..=255` are the base tokens; the ID equals the byte value, so
//!   token `104` means the single byte `0x68` ("h").
//! - Every merge rule `(left, right)` creates one new token. Rule number `i`
//!   (counting from 0) always creates the token with ID `256 + i`.
//! - The bytes of each token are stored once, in a table indexed by ID
//!   (`tokens`). Merged token bytes are simply `left` bytes followed by
//!   `right` bytes.
//!
//! Why integers and not strings such as `"<68><65>"`?
//!
//! - Integers are `Copy`: the hot loops (pair counting and merging) never
//!   allocate or clone, which is by far the biggest speed-up in training.
//! - A pair `(u32, u32)` is a cheap, fixed-size hash key.
//! - There is no text format to parse, so decoding is "concatenate bytes".
//! - A token can never be "missing from the vocabulary": every ID a rule
//!   creates already has an entry in the tokens table.
//!
//! ## Why Tokenization Matters
//!
//! Language models never see raw text—they see token IDs. This fundamental
//! transformation explains many "quirky" behaviors:
//!
//! - **Can't reliably count letters**: The model sees tokens like ["run", "ning"],
//!   not individual characters
//! - **Sensitive to spacing**: " hello" and "hello" tokenize differently
//! - **Struggles with rare words**: Uncommon words are broken into unfamiliar pieces
//! - **Reversal is hard**: Character-level operations are difficult when working with tokens
//!
//! ## Implementation Notes
//!
//! This implementation includes several optimizations for practical use:
//!
//! - **Integer tokens**: no string allocation or cloning in the hot loops
//! - **Parallel pair counting**: Uses Rayon to count pairs across CPU cores
//! - **Parallel encoding**: Large texts are split into chunks and encoded in parallel
//! - **In-place merging**: The corpus is rewritten in place with two indexes
//!   (read and write), so no second buffer is needed
//!
//! These optimizations are necessary to make training practical on CPU, but the
//! underlying algorithm is standard BPE.
//!
//! ## Reading Guide (suggested order for learners)
//!
//! 1. [`Tokenizer::default`]: the 256 starting tokens.
//! 2. [`train`](crate::tokenizer_functions::train): learn merge rules (read the five
//!    numbered STEPs in the loop, with a worked example).
//! 3. [`merge_pair_in_place`](crate::tokenizer_functions::merge_pair_in_place): the one function that both training and
//!    encoding use to apply a rule.
//! 4. [`encode`](crate::tokenizer_functions::encode): replay the rules on new text.
//! 5. [`decode`](crate::tokenizer_functions::decode): turn IDs back into text.
//! 6. Everything else (save/load, stats, analysis) is supporting code.
//!
//! Project convention (see `spec/index.md`): the logic lives in pure
//! functions, not `self` methods. You train, encode and decode with free
//! functions that take the tokenizer as an argument:
//! `train(&mut tokenizer, text, 512)`, `encode(&tokenizer, text)`,
//! `decode(&tokenizer, &ids)`. They are in `tokenizer_functions.rs`, along with
//! the rest ([`count_token_id_pairs`](crate::tokenizer_functions::count_token_id_pairs),
//! [`select_best_pair`](crate::tokenizer_functions::select_best_pair) and so on),
//! which take what they need as arguments and return a result. `Tokenizer`
//! itself keeps only what works on its own fields: construction (`default`,
//! `from_merges`), saving and loading, `vocab_size`, `stats` and
//! `analyze_vocabulary` (which prints).
//!
//! The whole round trip, in one view:
//!
//! ```text
//! text --UTF-8--> bytes [104, 105] --merge rules--> fewer, bigger IDs [256] --> model
//! IDs [256] --tokens table--> bytes [104, 105] --UTF-8--> text "hi"
//! ```
//!
//! ## Saved File Format
//!
//! Only the ordered merge rules are saved (plus a format version), because
//! the tokens table can be rebuilt from them. See [`Tokenizer::save`].

#[allow(unused_imports)]
use log::{Level, debug, error, info, log_enabled, warn};
use std::fs;
use std::path::Path;

// The pure functions, types and constants live in `tokenizer_functions.rs`.
use crate::tokenizer_functions::{
    BASE_TOKEN_COUNT, base_tokens, build_tokens_from_merges, merges_from_json, merges_into_json,
    vocabulary_report,
};

// Re-exported so that these types remain available as `tokenizer::TokenId` etc.
pub use crate::tokenizer_functions::{Token, TokenIdPair};

/// A Byte Pair Encoding tokenizer
///
/// The tokenizer holds an ordered list of merge rules learned during training,
/// plus a table giving the bytes of every token. The merges are applied in
/// order during encoding to convert text into token IDs.
///
/// # Invariants
///
/// These always hold, for a default, trained or loaded tokenizer:
///
/// 1. `tokens.len() == 256 + merges.len()`: one table entry per token.
/// 2. `tokens[b] == [b]` for `b` in `0..256`: the base tokens.
/// 3. For rule `i` = `(left, right)`: `left` and `right` are both `< 256 + i`
///    (a rule can only use tokens that already exist), and
///    `tokens[256 + i] == tokens[left] ++ tokens[right]`.
#[derive(Clone, Debug)]
pub struct Tokenizer {
    /// The content of each token, indexed by token ID.
    ///
    /// Entries `0..256` are the single-byte base tokens. Entry `256 + i` is
    /// the token created by merge rule `i`. The table's length is therefore
    /// the vocabulary size.
    ///
    /// This one table does the job that two maps (token -> ID and ID -> token)
    /// did in the string-based design: an ID is just an index.
    pub tokens: Vec<Token>,

    /// Merge rules in the order they were learned. Each rule joins two
    /// existing tokens into a new one. Applied in this order during encoding.
    /// The token created by rule `i` has ID `256 + i`, so IDs need no storage.
    pub merges: Vec<TokenIdPair>,
}

impl Default for Tokenizer {
    /// Create a default tokenizer with base vocabulary.
    ///
    /// Initializes the tokenizer with 256 base tokens representing each possible
    /// byte value (0-255). Token `n` stands for the single byte `n`.
    ///
    /// # Returns
    ///
    /// A new tokenizer with 256 base tokens and no merges
    fn default() -> Self {
        // `base_tokens` builds the 256 one-byte tokens: the entry at index `b`
        // is `[b]`, so the ID of a base token equals its byte value.
        let tokens: Vec<Token> = base_tokens();

        Self {
            tokens,
            merges: Vec::new(),
        }
    }
}

impl Tokenizer {
    /// Get the vocabulary size
    ///
    /// # Returns
    ///
    /// Number of tokens in the vocabulary: 256 base tokens plus one per merge
    /// rule, i.e. exactly `256 + number of merges`.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Save tokenizer to a JSON file
    ///
    /// Writes the merge rules (and a format version) to a JSON file for
    /// later loading. The tokens table is not saved because it is rebuilt from
    /// the rules, so the file is small and cannot contradict itself.
    ///
    /// Example file content:
    ///
    /// ```text
    /// {
    ///   "format_version": 1,
    ///   "merges": [[97, 97], [256, 256]]
    /// }
    /// ```
    ///
    /// Each pair is `[left_id, right_id]`; rule `i` creates token `256 + i`.
    ///
    /// Building the JSON text is the pure function [`merges_into_json`]; this
    /// method only adds the file write, which is the part with side effects.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the tokenizer
    ///
    /// # Returns
    ///
    /// Result indicating success or error
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // `ignore`: this crate is a binary, so doc tests cannot import it.
    /// let tokenizer = Tokenizer::default();
    /// tokenizer.save("tokenizer.json").expect("Failed to save");
    /// ```
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        // Pure step: rules -> JSON text.
        let json = merges_into_json(&self.merges)?;
        // Side effect: write the file.
        fs::write(path, json)?;
        Ok(())
    }

    /// Load tokenizer from a JSON file
    ///
    /// Reads a file written by [`save`](Self::save), checks it, and rebuilds
    /// the tokens table from the merge rules.
    ///
    /// Reading the file is the only side effect here; parsing and checking the
    /// text is the pure function [`merges_from_json`].
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the tokenizer file
    ///
    /// # Returns
    ///
    /// Result containing the loaded tokenizer or an error. Errors are returned
    /// (never panics) for: a missing or unreadable file, invalid JSON, a file
    /// in an old or different layout (including files saved before the
    /// integer-token redesign, which have no `format_version`; retrain those),
    /// an unsupported `format_version`, and rules that refer to tokens that do
    /// not exist yet.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // `ignore`: this crate is a binary, so doc tests cannot import it.
    /// let tokenizer = Tokenizer::load("tokenizer.json")
    ///     .expect("Failed to load tokenizer");
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        // Side effect: read the whole file. This can fail (missing or
        // unreadable file), and `?` converts the error into the boxed return type.
        let json = fs::read_to_string(path)?;

        // Pure steps: parse and check the JSON, then rebuild and validate the
        // tokenizer from its rules.
        let merges = merges_from_json(&json)?;
        Ok(Self::from_merges(merges)?)
    }

    /// Build a tokenizer from an ordered list of merge rules.
    ///
    /// Starts from the 256 base tokens and replays the rules to rebuild the
    /// tokens table, checking invariant 3 (see the struct docs) as it goes. A
    /// rule may only use tokens that exist when it is learned, i.e. both IDs
    /// must be smaller than `256 + rule_index`.
    ///
    /// This validation matters because the input may come from a file that was
    /// edited or damaged; without it, a bad ID would cause an out-of-bounds
    /// panic later, during encoding.
    ///
    /// The checking and rebuilding is done by the pure function
    /// [`build_tokens_from_merges`]; this constructor only stores the result.
    ///
    /// # Arguments
    ///
    /// * `merges` - the rules, in training order.
    ///
    /// # Errors
    ///
    /// Returns a message naming the first invalid rule.
    pub fn from_merges(merges: Vec<TokenIdPair>) -> Result<Self, String> {
        let tokens = build_tokens_from_merges(&merges)?;
        Ok(Self { tokens, merges })
    }

    /// Get statistics about the tokenizer
    ///
    /// # Returns
    ///
    /// TokenizerStats struct with vocabulary information
    pub fn stats(&self) -> TokenizerStats {
        TokenizerStats {
            vocab_size: self.tokens.len(),
            num_merges: self.merges.len(),
            // Constant: byte-level BPE always starts from 256 byte values.
            base_tokens: BASE_TOKEN_COUNT,
        }
    }

    /// Analyze vocabulary and display insights
    ///
    /// Prints detailed information about the tokenizer's vocabulary, including:
    /// - Token composition (base vs. learned)
    /// - Sample of learned tokens
    /// - Compression analysis on sample text
    /// - Example tokenizations
    ///
    /// This is useful for understanding what the tokenizer has learned.
    ///
    /// The printing is the only thing this method does itself. What to print
    /// is worked out by pure functions ([`learned_token_samples`](crate::tokenizer_functions::learned_token_samples),
    /// [`compression_ratio`](crate::tokenizer_functions::compression_ratio), [`token_strings`](crate::tokenizer_functions::token_strings)), which can be tested without
    /// capturing terminal output.
    ///
    /// # Arguments
    ///
    /// * `sample_text` - Text to use for compression analysis and examples
    pub fn analyze_vocabulary(&self, sample_text: &str) {
        // The report is built by the pure function `vocabulary_report`; this method
        // only prints it.
        print!("{}", vocabulary_report(self, sample_text));
    }
}

/// Statistics about a tokenizer's vocabulary
#[derive(Debug)]
pub struct TokenizerStats {
    /// Total vocabulary size (base tokens + learned merges)
    pub vocab_size: usize,
    /// Number of merge rules learned
    pub num_merges: usize,
    /// Number of base tokens (always 256 for byte-level BPE)
    pub base_tokens: usize,
}

// Tests. Most tests exercise round trips: whatever goes through `encode` must
// come back identical from `decode`, with or without learned merges. The rest
// check the invariants listed on `Tokenizer`, training
// behaviour, and save/load.
#[cfg(test)]
mod tests {
    // The pure functions are tested in `tokenizer_functions.rs`; the few tests
    // here that call them do so to check the Tokenizer's behavior.
    use super::*;
    use crate::tokenizer_functions::{TokenId, decode, encode, token_to_string, train};

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// Builds a tokenizer trained on `text` up to `vocab_size`.
    fn trained(text: &str, vocab_size: usize) -> Tokenizer {
        let mut tokenizer = Tokenizer::default();
        train(&mut tokenizer, text, vocab_size);
        tokenizer
    }

    /// A unique temp file path, so parallel tests never share a file.
    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("demo-nhs-llm-bpe-{}-{}.json", std::process::id(), name))
    }

    /// Asserts the three struct invariants (see `Tokenizer`).
    fn assert_invariants(tokenizer: &Tokenizer) {
        // 1. One table entry per token.
        assert_eq!(tokenizer.tokens.len(), 256 + tokenizer.merges.len());
        // 2. Base tokens are the single bytes.
        for byte in 0..256usize {
            assert_eq!(tokenizer.tokens[byte], vec![byte as u8]);
        }
        // 3. Each rule uses only earlier tokens, and its token is left ++ right.
        for (i, &(left, right)) in tokenizer.merges.iter().enumerate() {
            assert!((left as usize) < 256 + i, "rule {} uses a future left token", i);
            assert!((right as usize) < 256 + i, "rule {} uses a future right token", i);
            let mut expect = tokenizer.tokens[left as usize].clone();
            expect.extend_from_slice(&tokenizer.tokens[right as usize]);
            assert_eq!(tokenizer.tokens[256 + i], expect, "rule {} bytes", i);
        }
    }

    // ------------------------------------------------------------------
    // Default / base vocabulary
    // ------------------------------------------------------------------

    /// The default tokenizer has 256 tokens and the ID of each equals its byte value.
    #[test]
    fn test_default_base_vocabulary() {
        let tokenizer = Tokenizer::default();
        assert_eq!(tokenizer.vocab_size(), 256);
        assert!(tokenizer.merges.is_empty());
        assert_eq!(tokenizer.tokens[0], vec![0]);
        assert_eq!(tokenizer.tokens[0x41], vec![0x41]);
        assert_eq!(tokenizer.tokens[255], vec![255]);
        assert_invariants(&tokenizer);
    }

    /// `stats` reports sizes that agree with the tokenizer's own state.
    #[test]
    fn test_stats() {
        let tokenizer = trained("hello hello hello world", 280);
        let stats = tokenizer.stats();
        assert_eq!(stats.vocab_size, tokenizer.vocab_size());
        assert_eq!(stats.num_merges, tokenizer.merges.len());
        assert_eq!(stats.base_tokens, 256);
    }

    // ------------------------------------------------------------------
    // Training
    // ------------------------------------------------------------------

    /// A target of 256 or less means no work: the base vocabulary is already enough.
    #[test]
    fn test_train_target_at_or_below_base_is_noop() {
        for target in [0, 100, 256] {
            let tokenizer = trained("hello hello", target);
            assert_eq!(tokenizer.vocab_size(), 256);
            assert!(tokenizer.merges.is_empty());
        }
    }

    /// Empty text has no pairs, so nothing is learned and nothing panics.
    #[test]
    fn test_train_empty_text() {
        let tokenizer = trained("", 300);
        assert_eq!(tokenizer.vocab_size(), 256);
        assert!(tokenizer.merges.is_empty());
    }

    /// One byte has no pairs either.
    #[test]
    fn test_train_single_byte() {
        let tokenizer = trained("a", 300);
        assert!(tokenizer.merges.is_empty());
    }

    /// The first merge is the most frequent pair: in "aaabdaaabac", (a,a) occurs 4 times.
    #[test]
    fn test_train_first_merge_is_most_frequent_pair() {
        let tokenizer = trained("aaabdaaabac", 257);
        assert_eq!(tokenizer.merges.len(), 1);
        assert_eq!(tokenizer.merges[0], (0x61, 0x61));
        // The new token gets the first free ID, and its bytes are "aa".
        assert_eq!(tokenizer.tokens[256], b"aa".to_vec());
    }

    /// With equal counts, the pair whose bytes sort first wins.
    #[test]
    fn test_train_tie_breaks_on_smallest_pair() {
        // Pairs ab, bc, cd each occur once.
        let tokenizer = trained("abcd", 257);
        assert_eq!(tokenizer.merges[0], (0x61, 0x62));
    }

    /// Tie-breaking stays deterministic once merged tokens are in play, where
    /// several pairs may tie and some involve tokens with higher IDs.
    #[test]
    fn test_train_tie_break_is_deterministic_with_merged_tokens() {
        let text = "bcbc ad ad xy xy";
        let first = trained(text, 270);
        for _ in 0..5 {
            assert_eq!(trained(text, 270).merges, first.merges);
        }
        assert_invariants(&first);
    }

    /// Training reaches the requested size when the corpus has enough pairs.
    #[test]
    fn test_train_reaches_target_exactly() {
        // Highly repetitive text runs out of pairs after about 25 merges, so
        // the target must stay below that for this test.
        let text = "the cat sat on the mat. ".repeat(20);
        let tokenizer = trained(&text, 270);
        assert_eq!(tokenizer.vocab_size(), 270);
        assert_eq!(tokenizer.merges.len(), 14);
        assert_invariants(&tokenizer);
    }

    /// Training stops early when the corpus is fully merged, so tiny corpora
    /// cannot reach a large target.
    #[test]
    fn test_train_stops_early_on_tiny_corpus() {
        let tokenizer = trained("hello hello world", 10_000);
        assert!(tokenizer.vocab_size() > 256);
        assert!(tokenizer.vocab_size() < 1_000);
        assert_invariants(&tokenizer);
    }

    /// A later merge can build on an earlier one ("aa" then "aa"+"aa").
    #[test]
    fn test_train_merges_build_on_earlier_merges() {
        let tokenizer = trained("aaaa", 258);
        assert_eq!(tokenizer.merges.len(), 2);
        // Rule 1 joins token 256 ("aa") with itself.
        assert_eq!(tokenizer.merges[1], (256, 256));
        assert_eq!(tokenizer.tokens[257], b"aaaa".to_vec());
    }

    /// The structural invariants hold after training on varied text.
    #[test]
    fn test_train_preserves_invariants() {
        for text in ["", "a", "abcabcabc abcabc", "日本語 日本語", "hello hello world"] {
            assert_invariants(&trained(text, 400));
        }
    }

    /// Training is deterministic: two runs on the same text learn identical merges,
    /// even though pair counts live in a randomly-ordered HashMap.
    #[test]
    fn test_train_is_deterministic() {
        let text = "the cat sat on the mat with the hat and the bat";
        let a = trained(text, 300);
        let b = trained(text, 300);
        assert_eq!(a.merges, b.merges);
    }

    /// Training again with a larger target continues where it stopped, and gives
    /// the same rules as training straight to the larger target.
    #[test]
    fn test_train_can_be_resumed() {
        let text = "the cat sat on the mat. ".repeat(20);
        let straight = trained(&text, 290);

        let mut resumed = trained(&text, 270);
        assert_eq!(resumed.vocab_size(), 270);
        train(&mut resumed, &text, 290);

        assert_eq!(resumed.merges, straight.merges);
        assert_invariants(&resumed);
    }

    /// Training again with the same (or smaller) target changes nothing.
    #[test]
    fn test_train_twice_with_same_target_is_noop() {
        let text = "the cat sat on the mat. ".repeat(20);
        let mut tokenizer = trained(&text, 280);
        let before = tokenizer.merges.clone();
        train(&mut tokenizer, &text, 280);
        train(&mut tokenizer, &text, 260);
        assert_eq!(tokenizer.merges, before);
    }

    /// Merges are applied left to right without overlap.
    #[test]
    fn test_encode_merges_do_not_overlap() {
        let tokenizer = trained("aaaa", 258);
        // "aaa" -> [aa, a]; the second rule (aa,aa) cannot match.
        assert_eq!(encode(&tokenizer, "aaa"), vec![256, 0x61]);
        // "aaaa" -> [aa, aa] -> [aaaa]
        assert_eq!(encode(&tokenizer, "aaaa"), vec![257]);
    }

    /// Training on repetitive text makes the same text encode shorter than one token per byte.
    #[test]
    fn test_trained_tokenizer_compresses() {
        let text = "the cat sat on the mat. ".repeat(20);
        let tokenizer = trained(&text, 320);
        let ids = encode(&tokenizer, &text);
        assert!(ids.len() < text.len(), "{} tokens vs {} bytes", ids.len(), text.len());
    }

    /// With no merges learned, every byte is its own token and its ID is the byte value.
    #[test]
    fn test_encode_untrained_uses_only_base_ids() {
        let tokenizer = Tokenizer::default();
        let ids = encode(&tokenizer, "hi");
        assert_eq!(ids, vec![0x68, 0x69]);
    }

    /// Encoding the same text twice gives the same IDs.
    #[test]
    fn test_encode_is_stable() {
        let tokenizer = trained("one two three one two three", 300);
        assert_eq!(encode(&tokenizer, "one two"), encode(&tokenizer, "one two"));
    }

    /// Cloning gives an independent copy: training one does not change the other.
    #[test]
    fn test_clone_is_independent() {
        let original = Tokenizer::default();
        let mut copy = original.clone();
        train(&mut copy, "hello hello hello", 300);
        assert_eq!(original.vocab_size(), 256);
        assert!(copy.vocab_size() > 256);
    }

    // ------------------------------------------------------------------
    // Encode / decode
    // ------------------------------------------------------------------

    /// A simple string round-trips with only base tokens.
    #[test]
    fn test_decode_basic() {
        let tokenizer = Tokenizer::default();
        let text = "hello";
        let ids = encode(&tokenizer, text);
        assert_eq!(decode(&tokenizer, &ids), text);
    }

    /// A range of tricky strings round-trip with an untrained tokenizer.
    #[test]
    fn test_encode_decode_roundtrip() {
        let tokenizer = Tokenizer::default();

        let test_cases = vec![
            "hello",
            "Hello, world!",
            "To be, or not to be",
            "123 456 789",
            "special chars: !@#$%^&*()",
            "newline\nand\ttab",
            "UTF-8: café, naïve, 日本語",
        ];

        for text in test_cases {
            let encoded = encode(&tokenizer, text);
            let decoded = decode(&tokenizer, &encoded);
            assert_eq!(decoded, text, "Failed roundtrip for: {}", text);
        }
    }

    /// Encode/decode still works after training adds merges.
    #[test]
    fn test_encode_decode_with_merges() {
        let tokenizer = trained("hello hello world world hello", 300);
        let test_text = "hello world";
        let encoded = encode(&tokenizer, test_text);
        assert_eq!(decode(&tokenizer, &encoded), test_text);
    }

    /// Every byte value is encodable and decodes back to itself.
    #[test]
    fn test_base_vocab_coverage() {
        let tokenizer = Tokenizer::default();
        for id in 0..256u32 {
            // Decoding a lone base token gives exactly that byte (lossy for
            // bytes >= 0x80, which are not valid UTF-8 on their own).
            let bytes_back = tokenizer.tokens[id as usize].clone();
            assert_eq!(bytes_back, vec![id as u8]);
        }
        // ASCII round-trips through text.
        for byte in 0u8..128 {
            let text = (byte as char).to_string();
            assert_eq!(decode(&tokenizer, &encode(&tokenizer, &text)), text);
        }
    }

    /// Decoding works on a tokenizer that has just been trained.
    #[test]
    fn test_decode_after_train() {
        let mut tokenizer = Tokenizer::default();
        let _ = decode(&tokenizer, &[104]);
        train(&mut tokenizer, "hello hello hello", 300);
        let ids = encode(&tokenizer, "hello");
        assert!(ids.iter().any(|&id| id >= 256)); // uses a learned token
        assert_eq!(decode(&tokenizer, &ids), "hello");
    }

    /// Vocabulary size grows with training but never beyond the target.
    #[test]
    fn test_vocab_size() {
        let tokenizer = Tokenizer::default();
        assert_eq!(tokenizer.vocab_size(), 256);

        let tokenizer2 = trained("hello hello world", 512);
        assert!(tokenizer2.vocab_size() > 256);
        assert!(tokenizer2.vocab_size() <= 512);
    }

    /// Empty input round-trips to empty output.
    #[test]
    fn test_encode_decode_empty() {
        let tokenizer = trained("hello hello", 300);
        assert!(encode(&tokenizer, "").is_empty());
        assert_eq!(decode(&tokenizer, &[]), "");
    }

    /// Trained tokenizers round-trip text it never saw, including non-ASCII text.
    #[test]
    fn test_roundtrip_unseen_and_unicode_text_after_training() {
        let tokenizer = trained("hello hello world world", 300);
        for text in ["goodbye", "café naïve", "日本語のテキスト", "tab\tnew\nline", "🙂 emoji"] {
            let ids = encode(&tokenizer, text);
            assert_eq!(decode(&tokenizer, &ids), text, "roundtrip failed for {:?}", text);
        }
    }

    /// Training on non-ASCII text can learn merges across the bytes of one character.
    #[test]
    fn test_train_on_unicode_text_roundtrips() {
        let text = "日本語 日本語 日本語";
        let tokenizer = trained(text, 300);
        assert_eq!(decode(&tokenizer, &encode(&tokenizer, text)), text);
        assert!(encode(&tokenizer, text).len() < text.len());
    }

    /// Unknown IDs are skipped when decoding rather than causing an error.
    #[test]
    fn test_decode_skips_unknown_ids() {
        let tokenizer = Tokenizer::default();
        // 104 = 'h', 105 = 'i'; 999999 is not in the vocabulary.
        assert_eq!(decode(&tokenizer, &[104, 999_999, 105]), "hi");
        // The largest possible ID must not overflow or panic either.
        assert_eq!(decode(&tokenizer, &[TokenId::MAX]), "");
    }

    /// A multi-byte character split across tokens is repaired by decoding all
    /// the bytes together.
    #[test]
    fn test_decode_joins_bytes_across_tokens() {
        let tokenizer = Tokenizer::default();
        // "é" is the two bytes 0xC3 0xA9, here as two separate base tokens.
        assert_eq!(decode(&tokenizer, &[0xc3, 0xa9]), "é");
    }

    /// A lone byte of a multi-byte character decodes to the replacement character.
    #[test]
    fn test_decode_partial_utf8_is_lossy() {
        let tokenizer = Tokenizer::default();
        assert_eq!(decode(&tokenizer, &[0xc3]), "\u{FFFD}");
    }

    /// `token_to_string` returns the text of a token, or `None` if unknown.
    #[test]
    fn test_token_to_string() {
        let tokenizer = trained("aaaa", 258);
        assert_eq!(token_to_string(&tokenizer.tokens, 0x61).as_deref(), Some("a"));
        assert_eq!(token_to_string(&tokenizer.tokens, 256).as_deref(), Some("aa"));
        assert_eq!(token_to_string(&tokenizer.tokens, 257).as_deref(), Some("aaaa"));
        assert_eq!(token_to_string(&tokenizer.tokens, 258), None);
    }

    /// Texts above the parallel threshold still round-trip, across many chunk
    /// boundaries, with multi-byte characters that must not be split.
    #[test]
    fn test_large_text_parallel_encode_roundtrip() {
        let tokenizer = trained(&"héllo wörld ".repeat(10), 300);
        let text = "héllo wörld ".repeat(20_000); // about 280 KB, above 2 x 100 KB
        assert!(text.len() > 200_000);
        let ids = encode(&tokenizer, &text);
        assert_eq!(decode(&tokenizer, &ids), text);
    }

    /// The parallel path keeps the chunks in order.
    #[test]
    fn test_large_text_parallel_encode_preserves_order() {
        let tokenizer = Tokenizer::default();
        // Untrained: ID == byte value, so the output must equal the input bytes.
        let text = "0123456789".repeat(30_000); // 300 KB
        let ids = encode(&tokenizer, &text);
        let expect: Vec<TokenId> = text.bytes().map(TokenId::from).collect();
        assert_eq!(ids, expect);
    }

    // ------------------------------------------------------------------
    // Save / load
    // ------------------------------------------------------------------

    /// Saving then loading gives an equivalent tokenizer that encodes and decodes identically.
    #[test]
    fn test_save_load_roundtrip() {
        let path = temp_path("roundtrip");
        let tokenizer = trained("hello hello world world hello", 300);
        tokenizer.save(&path).unwrap();
        let loaded = Tokenizer::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(loaded.merges, tokenizer.merges);
        // The tokens table is not saved; it must be rebuilt identically.
        assert_eq!(loaded.tokens, tokenizer.tokens);
        assert_invariants(&loaded);
        let ids = encode(&tokenizer, "hello world");
        assert_eq!(encode(&loaded, "hello world"), ids);
        assert_eq!(decode(&loaded, &ids), "hello world");
    }

    /// The saved file holds only the version and the rules, not the derived tokens table.
    #[test]
    fn test_save_file_contents() {
        let path = temp_path("contents");
        let tokenizer = trained("aaaa", 258);
        tokenizer.save(&path).unwrap();
        let json = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["format_version"], 1);
        assert_eq!(value["merges"], serde_json::json!([[97, 97], [256, 256]]));
        assert!(!json.contains("tokens"));
    }

    /// Loading a file that does not exist is an error, not a panic.
    #[test]
    fn test_load_missing_file_is_error() {
        assert!(Tokenizer::load(temp_path("does-not-exist")).is_err());
    }

    /// Loading a file that is not valid tokenizer JSON is an error.
    #[test]
    fn test_load_invalid_json_is_error() {
        let path = temp_path("invalid");
        std::fs::write(&path, "this is not json").unwrap();
        let result = Tokenizer::load(&path);
        std::fs::remove_file(&path).unwrap();
        assert!(result.is_err());
    }

    /// A file in the old layout (no `format_version`) is refused with an error.
    #[test]
    fn test_load_old_layout_is_error() {
        let path = temp_path("old-layout");
        std::fs::write(&path, r#"{"vocab":{},"merges":[],"unknown_token":"<unknown>"}"#).unwrap();
        let result = Tokenizer::load(&path);
        std::fs::remove_file(&path).unwrap();
        assert!(result.is_err());
    }

    /// A different format version is refused, with the version in the message.
    #[test]
    fn test_load_wrong_version_is_error() {
        let path = temp_path("wrong-version");
        std::fs::write(&path, r#"{"format_version":99,"merges":[]}"#).unwrap();
        let result = Tokenizer::load(&path);
        std::fs::remove_file(&path).unwrap();
        let message = result.err().unwrap().to_string();
        assert!(message.contains("99"), "{}", message);
    }

    /// A rule that uses a token which does not exist yet is refused at load time,
    /// instead of causing an out-of-bounds panic later.
    #[test]
    fn test_load_rejects_rule_using_future_token() {
        let path = temp_path("future-token");
        std::fs::write(&path, r#"{"format_version":1,"merges":[[300,1]]}"#).unwrap();
        let result = Tokenizer::load(&path);
        std::fs::remove_file(&path).unwrap();
        assert!(result.is_err());
    }

    /// A rule may not refer to the token it is about to create (256 for rule 0).
    #[test]
    fn test_from_merges_rejects_self_reference() {
        assert!(Tokenizer::from_merges(vec![(256, 1)]).is_err());
        assert!(Tokenizer::from_merges(vec![(1, 256)]).is_err());
        // Rule 1 may use token 256, which rule 0 created.
        assert!(Tokenizer::from_merges(vec![(97, 97), (256, 1)]).is_ok());
    }

    /// A hand-written valid file loads and works.
    #[test]
    fn test_load_hand_written_file() {
        let path = temp_path("hand-written");
        std::fs::write(&path, r#"{"format_version":1,"merges":[[97,97]]}"#).unwrap();
        let loaded = Tokenizer::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(loaded.vocab_size(), 257);
        assert_eq!(encode(&loaded, "aa"), vec![256]);
    }

    // ------------------------------------------------------------------
    // Reporting
    // ------------------------------------------------------------------

    /// The report runs without panicking on trained and untrained tokenizers,
    /// with and without a sample text.
    #[test]
    fn test_analyze_vocabulary_does_not_panic() {
        Tokenizer::default().analyze_vocabulary("");
        let tokenizer = trained("To be, or not to be, that is the question", 300);
        tokenizer.analyze_vocabulary("");
        tokenizer.analyze_vocabulary("To be, or not to be");
    }
}
