//! Pure functions for the Byte Pair Encoding (BPE) tokenizer
//!
//! Project convention (see `spec/index.md`): prefer pure functions, not `self`
//! methods. Each function here takes everything it needs as arguments and
//! returns its result, so it can be read, tested and reused on its own,
//! and without a method call. The tokenizer operations, [`train`], [`encode`]
//! and [`decode`], are free functions that take a [`Tokenizer`] as their first
//! argument: `encode(&tokenizer, text)`, not `tokenizer.encode(text)`. The
//! `Tokenizer` type in `tokenizer.rs` keeps only what works on its own fields.
//!
//! Functions are public by default.
//!
//! ## What is in this file
//!
//! - The shared types ([`Token`], [`TokenId`], [`TokenIdPair`],
//!   [`TokenIdPairToCountMap`]) and constants ([`BASE_TOKEN_COUNT`],
//!   [`FORMAT_VERSION`]). They live here, rather than in `tokenizer.rs`,
//!   because the functions need them and a module should not depend on the
//!   module that depends on it. `tokenizer.rs` re-exports them.
//! - The on-disk shape of a saved tokenizer (`SavedTokenizer`, private).
//! - The functions, in seven groups:
//!
//!   1. The tokens table: [`base_tokens`], [`new_token_from_pair`],
//!      [`build_tokens_from_merges`], [`token_to_string`]
//!   2. Training: [`count_token_id_pairs`], [`select_best_pair`],
//!      [`merge_pair_in_place`], [`from_text_into_token_ids`]
//!   3. Encoding: [`apply_merges`], [`encode_sequential`],
//!      [`split_text_into_chunks`], [`encode`]
//!   4. Decoding: [`decode`]
//!   5. Saving and loading: [`merges_into_json`], [`merges_from_json`]
//!   6. Reporting: [`learned_token_samples`], [`compression_ratio`],
//!      [`token_strings`]
//!   7. Sizing and sampling: [`text_sample`], [`chunk_size`]
//!   8. The training and analysis operations: [`train`] (learn the merge
//!      rules) and [`vocabulary_report`] (the analysis text)

use crate::tokenizer::Tokenizer;
use log::info;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// === Type aliases and constants ===
// These give the data structures below domain names, so signatures read as
// intent rather than nested generics.

/// The content a token stands for, e.g. `[104]` for "h" or `[104, 105]` for "hi".
pub type Token = Vec<u8>;

/// A token's integer ID, which is what a language model actually consumes.
///
/// `u32` is plenty (4 billion tokens) and half the size of `usize` on 64-bit
/// machines, which matters when the whole corpus is held as a list of IDs.
pub type TokenId = u32;

/// Two adjacent token IDs. Also the form of one merge rule: "join left and right".
pub type TokenIdPair = (TokenId, TokenId);

/// Pair -> how many times it occurs. This is what pair counting produces in `train`.
pub type TokenIdPairToCountMap = HashMap<TokenIdPair, usize>;

/// Number of base tokens: one for each possible byte value (0-255).
pub const BASE_TOKEN_COUNT: usize = 256;

/// Version number written into saved files.
///
/// Increase this whenever the file layout changes, so that `load` can refuse
/// an incompatible file with a clear message rather than misreading it.
pub const FORMAT_VERSION: u32 = 1;

/// The on-disk shape of a tokenizer.
///
/// Deliberately smaller than [`Tokenizer`]: the tokens table is
/// derived data and is rebuilt from `merges` on load. Saving less means a
/// saved file can never contradict itself.
#[derive(Serialize, Deserialize)]
struct SavedTokenizer {
    /// Layout version; see [`FORMAT_VERSION`].
    format_version: u32,

    /// The merge rules, in training order.
    merges: Vec<TokenIdPair>,
}

// --- 1. The tokens table ---

/// The 256 base tokens: token `b` stands for the single byte `b`.
///
/// # Returns
///
/// A table of 256 one-byte tokens, indexed by token ID.
pub fn base_tokens() -> Vec<Token> {
    // `0..=255u8` yields every possible byte. For each byte `b` we store
    // the one-element byte list `[b]`. Because the table is indexed by ID
    // and we push in order, the entry at index `b` is `[b]`: the ID of a
    // base token equals its byte value.
    (0..=255u8).map(|byte| vec![byte]).collect()
}

/// The bytes of the token that merging `pair` would create: left bytes, then right bytes.
///
/// Example: with `tokens[104] = "h"` and `tokens[105] = "i"`, the pair
/// `(104, 105)` gives the bytes of "hi".
///
/// # Arguments
///
/// * `tokens` - the tokens table, indexed by token ID.
/// * `pair` - `(left, right)` IDs; both must be valid indexes into `tokens`.
///
/// # Panics
///
/// Panics if either ID is outside the table.
pub fn new_token_from_pair(tokens: &[Token], pair: TokenIdPair) -> Token {
    // `concat` on a slice of byte lists joins them into one new list.
    [
        tokens[pair.0 as usize].as_slice(),
        tokens[pair.1 as usize].as_slice(),
    ]
    .concat()
}

/// Rebuild the tokens table from an ordered list of merge rules, validating each rule.
///
/// Starts from the 256 base tokens and replays the rules, checking invariant 3
/// (see the [`Tokenizer`] docs) as it goes: a rule may only use tokens that exist
/// when it is learned, i.e. both IDs must be smaller than `256 + rule_index`.
///
/// # Arguments
///
/// * `merges` - the rules, in training order.
///
/// # Returns
///
/// The table, with `256 + merges.len()` entries.
///
/// # Errors
///
/// Returns a message naming the first invalid rule.
pub fn build_tokens_from_merges(merges: &[TokenIdPair]) -> Result<Vec<Token>, String> {
    // Begin with the 256 base tokens and no rules.
    let mut tokens = base_tokens();

    for (rule_idx, &(left, right)) in merges.iter().enumerate() {
        // The newest token that exists so far has ID `256 + rule_idx - 1`,
        // so valid IDs are strictly below `256 + rule_idx`.
        let token_count_so_far = tokens.len();
        if left as usize >= token_count_so_far || right as usize >= token_count_so_far {
            return Err(format!(
                "invalid merge rule {} ({}, {}): refers to a token that does not exist yet (only {} tokens exist at that point)",
                rule_idx, left, right, token_count_so_far
            ));
        }

        // Same construction as STEP 4 of `train`: bytes of left then right.
        let new_token: Token = new_token_from_pair(&tokens, (left, right));
        tokens.push(new_token);
    }

    Ok(tokens)
}

/// The text a single token stands for, or `None` for an unknown ID.
///
/// Used for logs and reports. A token that holds only part of a multi-byte
/// character shows as `U+FFFD`, because its bytes are not valid UTF-8 on
/// their own.
///
/// # Arguments
///
/// * `tokens` - the tokens table, indexed by token ID.
/// * `id` - the token ID to look up.
pub fn token_to_string(tokens: &[Token], id: TokenId) -> Option<String> {
    tokens
        .get(id as usize)
        .map(|token| String::from_utf8_lossy(token).into_owned())
}

// --- 2. Training ---

/// Count how often each adjacent pair of token IDs occurs (STEP 1 of `train`).
///
/// Example: `[a, a, b]` contains the pairs `(a,a)` and `(a,b)`, one each.
///
/// This is the performance bottleneck of training, so it runs in parallel:
/// the list is split into chunks, each chunk is counted on its own thread,
/// and the partial counts are added together.
///
/// # Arguments
///
/// * `token_ids` - the corpus as a flat list of token IDs.
///
/// # Returns
///
/// A map from each pair that occurs to its number of occurrences. Empty for
/// a list of fewer than 2 tokens.
pub fn count_token_id_pairs(token_ids: &[TokenId]) -> TokenIdPairToCountMap {
    // At least 50K tokens per chunk, or divide by thread count.
    // The local variable shadows the function of the same name; the
    // call on the right-hand side still resolves to the function.
    // Small chunks would cost more in scheduling than they save.
    let chunk_size = chunk_size(token_ids.len(), 50_000);

    // `token_ids` is only read here, so it is shared freely across threads.
    token_ids
        // Yields immutable parallel chunks i.e. data slices of a given size
        .par_chunks(chunk_size)
        .enumerate()
        // Rayon `fold` (takes a constructor, `TokenIdPairToCountMap::new`,
        // not a value): each worker thread builds its own private HashMap
        // of pair counts over the chunks it is given, so no locking is
        // needed. The closure receives the accumulator (`local_counts`)
        // and one item, which is a `(chunk_idx, chunk)` tuple from
        // `enumerate()` above. Note: rayon may call `fold` several
        // times per thread, so this produces many partial maps, not
        // exactly one per thread.
        .fold(TokenIdPairToCountMap::new, |mut local_counts, (chunk_idx, chunk)| {
            // Count pairs within this chunk.
            // `windows(2)` yields every overlapping pair of adjacent tokens:
            // [a, b, c] -> [a, b], [b, c].
            // The pair is two `u32`s, which are `Copy`, so building the key
            // costs nothing: no allocation and no `.clone()`.
            for window in chunk.windows(2) {
                let pair: TokenIdPair = (window[0], window[1]);
                // Insert a zero count if absent, then increment.
                *local_counts.entry(pair).or_insert(0) += 1;
            }

            // Handle chunk boundaries: count the pair spanning to next chunk.
            // `windows(2)` never sees a pair that straddles two chunks, so
            // the last token of this chunk is paired with the first token of
            // the next chunk. Each chunk owns the boundary on its right side,
            // so every boundary pair is counted exactly once.
            // `chunk_idx * chunk_size` is the chunk's start offset in `token_ids`
            // (every chunk but the last has exactly `chunk_size` items), and
            // adding `chunk.len()` gives the index of the next chunk's first token.
            let next_chunk_start = chunk_idx * chunk_size + chunk.len();

            // Two things must both be present for a boundary pair:
            //   - `chunk.last()`: `None` only for an empty chunk (should not occur)
            //   - `token_ids.get(next_chunk_start)`: `None` for the final chunk,
            //     which has no successor, so no pair is added there.
            // Matching both in one `if let` also protects against indexing
            // past the end: `get` returns `None` instead of panicking.
            if let (Some(&last), Some(&next)) = (chunk.last(), token_ids.get(next_chunk_start)) {
                *local_counts.entry((last, next)).or_insert(0) += 1;
            }

            // Return the accumulator so rayon can pass it to the next fold step.
            local_counts
        })
        // Rayon `reduce`: combine the partial maps from `fold` into one.
        // `HashMap::new` is the identity (empty) value; it is a function
        // because rayon needs a fresh one for each parallel branch.
        .reduce(HashMap::new, |mut a, b| {
            // Merge pair counts from all chunks by summing counts per pair.
            for (pair, count) in b {
                *a.entry(pair).or_insert(0) += count;
            }
            a
        })
}

/// Choose the pair to merge next (STEP 3 of `train`), or `None` if there are no pairs.
///
/// Example: counts `{(a,a): 4, (a,b): 2}` -> the winner is `(a,a)` with count 4.
///
/// # Deterministic tie-breaking
///
/// HashMap iteration order is random, so when two pairs have the same
/// count we need a fixed rule to pick one; otherwise training would
/// learn different merges from run to run.
/// Winner = highest count. On a tie, the pair whose token BYTES sort
/// first wins (left token's bytes, then right token's bytes). We
/// compare bytes rather than IDs so that the choice does not depend
/// on the order in which IDs happened to be created.
/// If even the bytes are equal (two different IDs can stand for the
/// same bytes), the smaller IDs win, so the order is total.
/// A single O(P) `max_by` scan finds the winner for P distinct pairs,
/// with no need to sort them all.
///
/// # Arguments
///
/// * `counts` - pair counts, as returned by [`count_token_id_pairs`].
/// * `tokens` - the tokens table, used to compare the bytes behind the IDs.
///
/// # Returns
///
/// The winning pair and its count.
pub fn select_best_pair(counts: TokenIdPairToCountMap, tokens: &[Token]) -> Option<(TokenIdPair, usize)> {
    counts.into_iter().max_by(|a, b| {
        // Look up the bytes behind a pair of IDs; `as usize` because
        // slice indexes must be `usize`.
        let bytes_of = |pair: &TokenIdPair| (&tokens[pair.0 as usize], &tokens[pair.1 as usize]);
        a.1.cmp(&b.1) // Higher count wins
            // Ties: reversed comparison, because `max_by` keeps the
            // GREATER item and we want the SMALLER bytes to win.
            .then_with(|| bytes_of(&b.0).cmp(&bytes_of(&a.0)))
            .then_with(|| b.0.cmp(&a.0)) // Last resort: smaller IDs win
    })
}

/// Replace every occurrence of `pair` in `token_ids` with `new_token_id`, in place.
///
/// This is the single place where a merge rule is applied; both `train`
/// (STEP 5) and `encode` (via `apply_merges`) call it.
///
/// # How it works: two indexes over one list
///
/// The output can never be longer than the input (each match turns 2 tokens
/// into 1), so we can write the result into the same list we are reading:
///
/// - `read` walks forward over the original tokens.
/// - `write` marks where the next output token goes.
/// - `write <= read` always, so we never overwrite a token we have not read yet.
///
/// Worked example: pair `(a, a)`, new token `Z`, tokens `[a, a, a, b]`
///
/// ```text
/// read=0 write=0  token_ids[0..2] = (a,a) matches  -> write Z   -> [Z, a, a, b]  read=2 write=1
/// read=2 write=1  token_ids[2..4] = (a,b) no match -> copy a    -> [Z, a, a, b]  read=3 write=2
/// read=3 write=2  last token, no pair           -> copy b    -> [Z, a, b, b]  read=4 write=3
/// read=4: done. Cut the list to `write` = 3 tokens          -> [Z, a, b]
/// ```
///
/// Matches never overlap and are found left to right: `[a, a, a]` becomes
/// `[Z, a]`, not `[Z, Z]`.
///
/// # Arguments
///
/// * `token_ids` - the list to rewrite; it shrinks by one for each match.
/// * `pair` - the `(left, right)` IDs to look for.
/// * `new_token_id` - the ID that replaces each matched pair.
pub fn merge_pair_in_place(token_ids: &mut Vec<TokenId>, token_id_pair: TokenIdPair, new_token_id: TokenId) {
    let mut read = 0; // next token to examine
    let mut write = 0; // next free output slot

    while read < token_ids.len() {
        // `read + 1 < len` guarantees `token_ids[read + 1]` exists, with no
        // subtraction that could underflow on an empty list.
        if read + 1 < token_ids.len() && token_ids[read] == token_id_pair.0 && token_ids[read + 1] == token_id_pair.1 {
            // Match: one output token stands for two input tokens.
            token_ids[write] = new_token_id;
            read += 2; // skip both tokens in the pair
        } else {
            // No match: copy the token through unchanged. When nothing has
            // matched yet, `write == read` and this assignment is a no-op.
            token_ids[write] = token_ids[read];
            read += 1;
        }
        write += 1;
    }

    // Everything from `write` onward is leftover input; drop it.
    token_ids.truncate(write);
}

/// Convert from training text to byte-level tokens.
///
/// Each byte becomes one token whose ID equals the byte value, so "hi"
/// becomes `[104, 105]`.
///
/// The text is read as UTF-8 bytes, not characters, so a character such as
/// "é" (two bytes) yields two tokens. The output length therefore equals
/// `text.len()`.
///
/// # Arguments
///
/// * `text` - the text to convert.
///
/// # Returns
///
/// One token ID per byte, e.g. `"hi"` -> `[104, 105]`.
pub fn from_text_into_token_ids(text: &str) -> Vec<TokenId> {
    // `u32::from(u8)` is a lossless widening conversion (it cannot fail).
    text.bytes().map(TokenId::from).collect()
}

// --- 3. Encoding ---

/// Apply every learned merge rule, in order, to a list of token IDs.
///
/// The list is rewritten in place. Rule `i` produces the token with ID
/// `256 + i`, so the IDs do not need to be stored or looked up.
///
/// Used by [`encode_sequential`] (to encode new text) and by `train`
/// (to resume from earlier rules).
///
/// # Arguments
///
/// * `merges` - the rules, in training order.
/// * `token_ids` - IDs to rewrite; normally the base byte tokens of some text.
pub fn apply_merges(merges: &[TokenIdPair], token_ids: &mut Vec<TokenId>) {
    // Replay the rules in the order they were learned. The order is
    // essential: later rules combine tokens produced by earlier ones.
    // Cost is O(rules x tokens), the main reason encoding is slower than
    // it needs to be; production tokenizers use a priority queue or
    // merge-rank lookup instead.
    for (rule_idx, &pair) in merges.iter().enumerate() {
        // With fewer than 2 tokens left there are no pairs, and so no
        // later rule can match: stop early.
        if token_ids.len() < 2 {
            break;
        }
        let new_token_id = (BASE_TOKEN_COUNT + rule_idx) as TokenId;
        merge_pair_in_place(token_ids, pair, new_token_id);
    }
}

/// Encode text sequentially (non-parallel version)
///
/// Performs the actual encoding by applying merge rules. [`encode`] uses
/// it for small texts, and for each piece of a large text.
///
/// # Arguments
///
/// * `merges` - the rules, in training order.
/// * `text` - Input text to encode
///
/// # Returns
///
/// Vector of token IDs
pub fn encode_sequential(merges: &[TokenIdPair], text: &str) -> Vec<TokenId> {
    // === ENCODING, STEP BY STEP ===
    // Example: encode "hi" after training learned the rule (104, 105).
    //
    //   STEP 1  text -> byte tokens        "hi"        -> [104, 105]
    //   STEP 2  replay merge rules in order [104, 105]  -> [256]
    //
    // There is no STEP 3 (look the tokens up): the tokens already ARE IDs.
    // In the string-based design this needed a vocabulary lookup, which could
    // fail; with integer tokens there is nothing to look up and nothing to miss.
    //
    // Training and encoding share the same replacement code
    // (`merge_pair_in_place`); the difference is that encoding does not count
    // anything. It trusts the rules already learned.

    // STEP 1: text -> one token per byte, exactly as in training.
    let mut token_ids = from_text_into_token_ids(text);

    // STEP 2: replay the rules.
    apply_merges(merges, &mut token_ids);

    token_ids
}

/// Cut text into pieces of about `chunk_size_bytes` bytes, never splitting a character.
///
/// The pieces are borrowed slices of `text` (nothing is copied), in order, and
/// together they cover the whole text exactly: joining them gives back `text`.
/// Each piece is `chunk_size_bytes` long, except that a piece is extended by up
/// to 3 bytes to reach a UTF-8 character boundary, and the last piece is
/// whatever remains.
///
/// # Arguments
///
/// * `text` - the text to cut.
/// * `chunk_size_bytes` - the nominal piece length in bytes; must be at least 1.
///
/// # Panics
///
/// Panics if `chunk_size_bytes` is 0, which could never make progress.
pub fn split_text_into_chunks(text: &str, chunk_size_bytes: usize) -> Vec<&str> {
    assert!(chunk_size_bytes > 0, "chunk size must be at least 1 byte");

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < text.len() {
        // Calculate end position: nominally chunk_size_bytes ahead,
        // clamped to the end of the text for the final chunk.
        let mut end = (start + chunk_size_bytes).min(text.len());

        // Adjust to character boundary (can't split UTF-8 characters).
        // Slicing a `str` in the middle of a multi-byte character would
        // panic, so move forward to the next valid boundary (at most 3 bytes).
        while end < text.len() && !text.is_char_boundary(end) {
            end += 1;
        }

        chunks.push(&text[start..end]);

        // Move to next chunk
        start = end;
    }

    chunks
}

/// Encode text to token IDs
///
/// Converts text into a sequence of token IDs by first converting to byte-level
/// tokens, then applying learned merge rules in order.
///
/// # Arguments
///
/// * `tokenizer` - the tokenizer whose merge rules to apply
/// * `text` - Input text to encode
///
/// # Returns
///
/// Vector of token IDs
///
/// # Performance Optimization
///
/// For large texts (>200KB), this method splits the text into chunks and
/// encodes them in parallel, providing significant speedup on multi-core systems.
///
/// # Example
///
/// ```rust,ignore
/// // `ignore`: this crate is a binary, so doc tests cannot import it.
/// let tokenizer = Tokenizer::default();
/// let ids = encode(&tokenizer, "Hello, world!");
/// println!("Token IDs: {:?}", ids);
/// ```
pub fn encode(tokenizer: &Tokenizer, text: &str) -> Vec<TokenId> {
    let merges = &tokenizer.merges[..];

    // Threshold for parallel processing. Texts up to twice this size
    // take the sequential path, since thread overhead would outweigh gains.
    const CHUNK_SIZE: usize = 100_000; // bytes per chunk

    // Two paths, same result for ordinary text:
    //   small text -> encode in one go on this thread (simple, no overhead)
    //   large text -> split, encode the pieces on all CPU cores, glue back
    if text.len() > CHUNK_SIZE * 2 {
        // Large-text path, in three steps:
        //   STEP A  cut the text into ~100 KB pieces (on character boundaries)
        //   STEP B  encode every piece at the same time
        //   STEP C  concatenate the ID lists in the original order
        //
        // === PARALLEL ENCODING FOR LARGE TEXTS ===
        // Split text into non-overlapping chunks for parallel encoding
        //
        // Note: This means we don't apply merges across chunk boundaries.
        // This is acceptable because:
        // 1. Boundaries are rare (every 100KB)
        // 2. Impact on compression is negligible
        // 3. Correctness is guaranteed (no duplicate or missing tokens)

        // STEP A. `chunks` holds borrowed slices of `text`, so no text is copied.
        let chunks = split_text_into_chunks(text, CHUNK_SIZE);

        // STEP B. `par_iter` is rayon's parallel version of `iter`: the same
        // code, spread over a thread pool. Each call to `encode_sequential`
        // is independent (it only reads `merges`), so no locks are needed.
        // `par_iter` preserves order when collecting, so `encoded_chunks[i]`
        // corresponds to `chunks[i]`.
        let encoded_chunks: Vec<Vec<TokenId>> = chunks
            .par_iter()
            .map(|chunk| encode_sequential(merges, chunk))
            .collect();

        // STEP C. Example: chunk 0 -> [5, 9], chunk 1 -> [7] gives [5, 9, 7].
        let mut result = Vec::new();
        for chunk in encoded_chunks {
            result.extend_from_slice(&chunk);
        }
        result
    } else {
        // Small text: use sequential version (avoids parallel overhead)
        encode_sequential(merges, text)
    }
}

// --- 4. Decoding ---

/// Decode token IDs back to text
///
/// Converts a sequence of token IDs back into the original text by looking up
/// the bytes of each ID, joining them, and reading the result as UTF-8.
///
/// IDs that are not in the vocabulary are skipped rather than reported as
/// errors. Bytes that are not valid UTF-8 (for example a sequence that stops
/// in the middle of a multi-byte character) become the replacement
/// character `U+FFFD`.
///
/// # Arguments
///
/// * `tokenizer` - the tokenizer whose tokens table to read
/// * `ids` - Token IDs to decode
///
/// # Returns
///
/// Decoded text string
///
/// # Example
///
/// ```rust,ignore
/// // `ignore`: this crate is a binary, so doc tests cannot import it.
/// let tokenizer = Tokenizer::default();
/// let ids = encode(&tokenizer, "Hello!");
/// let text = decode(&tokenizer, &ids);
/// assert_eq!(text, "Hello!");
/// ```
pub fn decode(tokenizer: &Tokenizer, ids: &[TokenId]) -> String {
    let tokens = &tokenizer.tokens[..];

    // === DECODING, STEP BY STEP ===
    // Example: decode [256] when token 256 stands for the bytes [104, 105].
    //
    //   STEP 1  ID -> bytes          [256]         -> [104, 105]
    //   STEP 2  join all the bytes   (one list for the whole input)
    //   STEP 3  bytes -> text        [104, 105]    -> "hi"
    //
    // Decoding is exact: no information was lost by encoding, because every
    // merged token is just its bytes written next to each other.

    // STEPs 1 and 2: look up each ID's bytes and append them to one list.
    // `get` returns `None` for an ID past the end of the table; those IDs
    // are silently dropped. The lookup is a plain array index, O(1), so no
    // reverse map or cache is needed.
    let mut bytes: Vec<u8> = Vec::new();
    for &id in ids {
        if let Some(token) = tokens.get(id as usize) {
            bytes.extend_from_slice(token);
        }
    }

    // STEP 3: decode the whole byte sequence at once (not token by token).
    // A multi-byte UTF-8 character may be split across tokens, and is only
    // valid once all of its bytes are reassembled.
    // "Lossy" replaces any invalid sequence with U+FFFD instead of failing.
    String::from_utf8_lossy(&bytes).into_owned()
}

// --- 5. Saving and loading ---

/// Turn merge rules into the JSON text that `Tokenizer::save` writes.
///
/// Pure: it builds the text and does no file I/O. Pretty-printed, for
/// human-readable, diff-friendly files.
///
/// # Arguments
///
/// * `merges` - the rules, in training order.
///
/// # Returns
///
/// JSON such as `{"format_version": 1, "merges": [[97, 97]]}` (indented).
pub fn merges_into_json(merges: &[TokenIdPair]) -> Result<String, serde_json::Error> {
    // Copy the rules into the small on-disk struct. (A copy is fine: saving
    // is rare and the rule list is small compared to the corpus.)
    let saved = SavedTokenizer {
        format_version: FORMAT_VERSION,
        merges: merges.to_vec(),
    };
    serde_json::to_string_pretty(&saved)
}

/// Parse the JSON text of a saved tokenizer and return its merge rules.
///
/// Pure: it takes text and does no file I/O. It checks the JSON shape and the
/// `format_version`, but not the rules themselves; that is the job of
/// [`build_tokens_from_merges`].
///
/// # Arguments
///
/// * `json` - the contents of a file written by `Tokenizer::save`.
///
/// # Errors
///
/// Returns an error for invalid JSON, a different layout (including files
/// with no `format_version`), or an unsupported `format_version`.
pub fn merges_from_json(json: &str) -> Result<Vec<TokenIdPair>, Box<dyn std::error::Error>> {
    // Parse. A wrong shape (such as an old file layout) also fails here.
    let saved: SavedTokenizer = serde_json::from_str(json)?;

    // Refuse files from a different layout version instead of guessing.
    if saved.format_version != FORMAT_VERSION {
        return Err(format!(
            "unsupported tokenizer format_version {} (this program reads version {})",
            saved.format_version, FORMAT_VERSION
        )
        .into());
    }

    Ok(saved.merges)
}

// --- 6. Reporting (what `analyze_vocabulary` prints) ---

/// The first learned tokens, as `(id, text)`, leaving out the unprintable ones.
///
/// Looks at the first `max_count` learned tokens (IDs 256 and up, in creation
/// order). Tokens holding only part of a multi-byte character decode to
/// `U+FFFD`, and showing those would just be noise, so they are skipped, along
/// with empty and very long (over 20 bytes) ones, to keep the output tidy. The
/// result can therefore have fewer than `max_count` entries.
///
/// # Arguments
///
/// * `tokens` - the tokens table, indexed by token ID.
/// * `max_count` - how many learned tokens to look at.
pub fn learned_token_samples(tokens: &[Token], max_count: usize) -> Vec<(TokenId, String)> {
    let mut samples = Vec::new();
    for id in BASE_TOKEN_COUNT..tokens.len().min(BASE_TOKEN_COUNT + max_count) {
        // Convert the bytes back to text for display.
        if let Some(text) = token_to_string(tokens, id as TokenId) {
            if text.len() <= 20 && !text.is_empty() && !text.contains('\u{FFFD}') {
                samples.push((id as TokenId, text));
            }
        }
    }
    samples
}

/// Bytes represented per token: `byte_count / token_count`.
///
/// Example: 100 bytes encoded into 40 tokens -> 2.5. Higher is better. A
/// `token_count` of 0 gives infinity (or `NaN` for 0 bytes), so callers should
/// check for an empty sample first.
///
/// # Arguments
///
/// * `byte_count` - size of the original text, in bytes.
/// * `token_count` - number of tokens it encoded to.
pub fn compression_ratio(byte_count: usize, token_count: usize) -> f32 {
    // Both casts are to float to avoid integer division.
    byte_count as f32 / token_count as f32
}

/// The readable text of each token ID, with `"?"` for an unknown ID.
///
/// Example: IDs `[256, 98]` -> `["he", "b"]`. Joining with `|` shows where the
/// token boundaries are.
///
/// # Arguments
///
/// * `tokens` - the tokens table, indexed by token ID.
/// * `ids` - the IDs to show.
pub fn token_strings(tokens: &[Token], ids: &[TokenId]) -> Vec<String> {
    // `iter()` walks the IDs by reference, so the closure parameter is
    // `&id`; the pattern `|&id|` copies the number out of the reference.
    // Using `Vec<String>` as the target type tells `collect()` what to build.
    ids.iter()
        .map(|&id| {
            // ID -> bytes -> readable text, e.g. 300 -> " be". The table
            // lookup is O(1). It returns `None` for an unknown ID, which
            // should not happen for IDs we just produced.
            token_to_string(tokens, id)
                // An unknown ID shows as "?" instead of crashing.
                .unwrap_or_else(|| "?".to_string())
        })
        .collect()
}

// --- 8. Whole-operation functions (what the `Tokenizer` methods call) ---

/// Train on a text corpus.
///
/// Learns merge rules by iteratively finding and merging the most frequent
/// adjacent token pairs. This grows the vocabulary from its current size
/// (256 for a new tokenizer) up to the target vocabulary size.
///
/// Training can be continued: calling `train` again with a larger target
/// starts from the rules already learned and adds more. Calling it with a
/// target at or below the current size does nothing.
///
/// If the corpus runs out of pairs before the target is reached (for
/// example, a very short text), training stops early and the vocabulary
/// is smaller than requested.
///
/// Each round runs five STEPs, and each STEP is its own function: [`count_token_id_pairs`],
/// [`select_best_pair`], [`new_token_from_pair`] and [`merge_pair_in_place`]. The numbered STEP
/// comments in the body show how they connect.
///
/// # Arguments
///
/// * `tokenizer` - the tokenizer to train; its tables are extended in place
/// * `text` - Training corpus (typically several MB of text)
/// * `vocab_size` - Target vocabulary size (common values: 512, 1024, 2048, 4096)
///
/// # Example
///
/// ```rust,ignore
/// // `ignore`: this crate is a binary, so doc tests cannot import it.
/// let text = "hello world hello rust hello world";
/// let vocab_size = 300;
/// let mut tokenizer = Tokenizer::default();
/// train(&mut tokenizer, &text, vocab_size);
///
/// // Verify training worked
/// assert!(tokenizer.vocab_size() > 256);
/// ```
pub fn train(tokenizer: &mut Tokenizer, text: &str, vocab_size: usize) {
    let tokens = &mut tokenizer.tokens;
    let merges = &mut tokenizer.merges;

    // === TRAINING OVERVIEW (read this first) ===
    //
    // Goal: learn which neighbouring tokens appear together most often, and
    // give each such pair its own new token, so common text needs fewer tokens.
    //
    // Tiny worked example. Letters stand in for IDs so it is easy to read; the
    // real code uses numbers: "a" is token 97, so the pair is (97, 97).
    // Corpus: "aaabdaaabac"
    //
    //   Start:     a a a b d a a a b a c      (11 tokens, one per byte)
    //   Round 1:   most frequent pair is (a,a), seen 4 times
    //              new token Z = "aa" with ID 256; rule #0 is (a,a) -> 256
    //              a a a b d a a a b a c  becomes  Z a b d Z a b a c  (9 tokens)
    //   Round 2+:  count pairs again on the NEW, shorter list and repeat.
    //
    // Every round does the same five STEPs (marked below). Each round adds one
    // token to the vocabulary and one rule to `merges`, and the corpus
    // shrinks, so later rounds find bigger patterns made of earlier tokens.

    // --- SETUP ---

    // How many tokens exist now: 256 for a new tokenizer, more if resuming.
    let current_vocab_size = tokens.len();

    // If the target is not larger than what we have, there is nothing to learn:
    // BPE only ever adds tokens. This also protects the subtraction below from
    // underflowing (usize is unsigned).
    if vocab_size <= current_vocab_size {
        return;
    }

    // Plain `{}` format string: values are separated by commas in the text itself.
    info!(
        "Training Tokenizer: starting vocab size: {}, target vocab size: {}, corpus size: {} bytes",
        current_vocab_size,
        vocab_size,
        text.len()
    );

    // Each merge adds exactly one token, so we need (target - current) merges.
    let num_merges = vocab_size - current_vocab_size;

    // Turn the text into one token per byte, then apply any rules we already
    // have. For a new tokenizer there are none, so this is just the bytes.
    // For a resumed run it puts the corpus in exactly the state the earlier
    // training left it in, so new rounds continue where the old ones stopped.
    // `token_ids` is the whole corpus as a flat list of IDs; each merge below
    // rewrites it in place, so it shrinks over time.
    let mut token_ids: Vec<TokenId> = from_text_into_token_ids(text);
    apply_merges(merges, &mut token_ids);

    // Learn merges iteratively. Each pass: count pairs, pick the most
    // frequent, record it as a merge rule, then apply it to the corpus.
    // The order of the rules matters: `encode` replays them in this order.
    for merge_idx in 0..num_merges {
        // --- STEP 1 of 5: COUNT every adjacent pair (in parallel) ---
        // Example: tokens [a, a, b] contain the pairs (a,a) and (a,b), one each.
        // This is the performance bottleneck, so it runs in parallel; see
        // `count_token_id_pairs` for how the work is split across threads.
        let token_id_pair_to_count_map: TokenIdPairToCountMap = count_token_id_pairs(&token_ids);

        // --- STEP 2 of 5: STOP if there is nothing left to merge ---
        // A list of 0 or 1 tokens has no pairs. That only happens when the whole
        // corpus has been merged into a single token, so we exit the loop early.
        // The final vocabulary will then be smaller than `vocab_size`.
        if token_id_pair_to_count_map.is_empty() {
            break;
        }

        // --- STEP 3 of 5: PICK the winning pair ---
        // Example: counts {(a,a): 4, (a,b): 2} -> the winner is (a,a).
        // The rule, including the deterministic tie-breaking, lives in
        // `select_best_pair`. The map is non-empty here because the empty
        // case already broke out of the loop above.
        let (best_pair, count) = select_best_pair(token_id_pair_to_count_map, tokens)
            .expect("pair count map is not empty");

        // --- STEP 4 of 5: CREATE the new token and RECORD the rule ---
        // Example: winner (97, 97) -> new token 256 whose bytes are [97, 97] ("aa").
        //
        // The new token's ID is the current table length, so IDs stay dense
        // and later merges get higher IDs than earlier ones. It also equals
        // `256 + merges.len()`, which is how `encode` finds the ID of rule `i`
        // without storing it (invariant 3 in the struct docs).
        let new_token_id = tokens.len() as TokenId;

        // The new token's bytes are the left token's bytes then the right's.
        let new_token: Token = new_token_from_pair(tokens, best_pair);

        // Record the table entry and the rule together, so the invariant
        // `tokens.len() == 256 + merges.len()` is restored immediately.
        //
        // Note: two different rules can produce the same bytes, e.g.
        // ((ab), c) and (a, (bc)) both give "abc". Each still gets its own ID.
        // That wastes one vocabulary slot but is harmless: encoding replays
        // the rules in the same order as training, so it makes the same
        // choices training did, and decoding only needs the bytes.
        tokens.push(new_token);
        merges.push(best_pair);

        // --- STEP 5 of 5: APPLY the rule to the whole corpus ---
        // Example: [a, a, a, b] with rule (a,a) -> [aa, a, b].
        // Shared with `encode`; see `merge_pair_in_place` for the details.
        merge_pair_in_place(&mut token_ids, best_pair, new_token_id);

        // Progress logging every 50 merges (and on the first one), so long
        // training runs show signs of life without flooding the log.
        // `merge_idx + 1` is shown so the count is human-friendly (1-based).
        if merge_idx % 50 == 0 {
            info!(
                "  Merge {}/{}: {:?} + {:?} -> {:?} (count: {}) -> vocab size: {}",
                merge_idx + 1,
                num_merges,
                // Show the pair as readable text instead of two bare numbers.
                token_to_string(tokens, best_pair.0).unwrap_or_default(),
                token_to_string(tokens, best_pair.1).unwrap_or_default(),
                token_to_string(tokens, new_token_id).unwrap_or_default(),
                count,
                tokens.len()
            );
        }
    }

    info!(
        "Training complete: vocab size: {}, merges: {}.",
        tokens.len(),
        merges.len(),
    );
}

/// Build the vocabulary analysis report as text
///
/// This is where `Tokenizer::analyze_vocabulary` gets its report; that method
/// prints the string this returns. Returning the text (instead of printing it)
/// keeps the function free of side effects, and lets tests check the output.
/// See the method for what the four parts show.
///
/// # Arguments
///
/// * `tokenizer` - the tokenizer to describe (its tokens table, and its merge rules
///   for encoding the sample and example phrases)
/// * `sample_text` - text for the compression analysis (empty skips that part)
///
/// # Returns
///
/// The full report, one `\n`-terminated line at a time.
pub fn vocabulary_report(tokenizer: &Tokenizer, sample_text: &str) -> String {
    let tokens = &tokenizer.tokens[..];

    // `outln!` works like `println!` but appends to `report` instead of printing,
    // so the whole analysis is built as one string and returned.
    let mut report = String::new();
    macro_rules! outln {
        ($($arg:tt)*) => {{
            report.push_str(&format!($($arg)*));
            report.push('\n');
        }};
    }

    // === ANALYSIS OVERVIEW ===
    // This method is a read-only report; it changes nothing. It answers
    // four questions about a trained tokenizer, one section each:
    //   PART 1  How many tokens are base bytes vs. learned merges?
    //   PART 2  What do the first learned tokens look like as text?
    //   PART 3  How well does it compress a sample of text?
    //   PART 4  How does it split a few example phrases?
    // Everything is printed with `println!`, because the output is meant for
    // a person at a terminal; use `stats()` if you need the numbers in code.
    outln!("\n=== Vocabulary Analysis ===\n");

    // --- PART 1 of 4: token composition ---
    // Every token is either one of the 256 base tokens or a learned merge,
    // so merges = total - 256. The base tokens are exactly IDs 0..256.
    let base_tokens = BASE_TOKEN_COUNT;
    let merged_tokens = tokens.len() - base_tokens;
    outln!("Token Composition:");
    outln!("  Base tokens (bytes): {}", base_tokens);
    outln!("  Learned merges: {}", merged_tokens);
    outln!("  Total vocabulary: {}\n", tokens.len());

    // --- PART 2 of 4: sample of learned tokens ---
    // Example output line:  [256] "e "   (token ID 256 stands for the text "e ")
    // Learned tokens are IDs 256 and up, already in creation (training)
    // order: early, most-frequent merges first. No sorting is needed
    // because the ID *is* the position in the table.
    // `learned_token_samples` also drops tokens that would only be noise
    // (partial characters, very long tokens).
    outln!("Sample of Learned Tokens (first 30):");
    for (id, text) in learned_token_samples(tokens, 30) {
        outln!("  [{}] \"{}\"", id, text);
    }

    // --- PART 3 of 4: compression on a sample ---
    // Idea: if the tokenizer learned useful merges, the same text needs fewer
    // tokens than the one-token-per-byte starting point.
    //   Example: 100 bytes encoded into 40 tokens -> ratio 100 / 40 = 2.5x.
    // A ratio near 1.0x means the merges aren't helping on this text, for
    // instance because the sample is very different from the training corpus.
    // Skipped for an empty sample, which would otherwise divide by zero.
    if !sample_text.is_empty() {
        outln!("\nCompression Analysis (on sample):");
        // Limit to the first 10,000 characters to keep analysis quick.
        let sample_chars: String = sample_text.chars().take(10000).collect();
        let token_ids = encode(tokenizer, &sample_chars);
        // `len()` counts bytes, not characters, so for non-ASCII text the two
        // differ. The ratio below is "bytes per token", the usual measure
        // for byte-level BPE, and is labelled as such.
        let byte_count = sample_chars.len();
        let token_count = token_ids.len();
        // Higher is better: more input bytes represented per token.
        let ratio = compression_ratio(byte_count, token_count);

        outln!("  Sample size: {} bytes", byte_count);
        outln!("  Token count: {} tokens", token_count);
        outln!("  Compression ratio: {:.2}x", ratio);
        outln!("  Avg bytes per token: {:.1}", ratio);
    }

    // --- PART 4 of 4: example tokenizations ---
    // Seeing real splits is the best way to build intuition. Expected shape:
    //   "To be, or not to be" -> 9 tokens: [To| be|,| or| not| to| be]
    // The `|` marks token boundaries. Common words tend to be one token
    // (often with their leading space); rare words break into pieces.
    // These are hard-coded Shakespeare lines (the original corpus this code
    // was written for); replace them with phrases from your own corpus for
    // meaningful output here.
    outln!("\nExample Tokenizations:");
    let examples = vec![
        "To be, or not to be",
        "Romeo and Juliet",
        "Wherefore art thou",
        "The quality of mercy",
    ];

    // For each example phrase we do three things:
    //   1. encode it to IDs,
    //   2. turn every ID back into readable text, one token at a time,
    //   3. print the pieces separated by `|`.
    for example in examples {
        // Step 1: text -> token IDs, e.g. "to be" -> [300, 301, 98, 101]
        // (the numbers here are illustrative).
        let token_ids = encode(tokenizer, example);

        // Step 2: IDs -> one readable string per token (an unknown ID shows
        // as "?" instead of crashing).
        let token_strs: Vec<String> = token_strings(tokens, &token_ids);

        // Step 3: print, e.g.
        //   "To be" -> 2 tokens: [To| be]
        // `join("|")` glues the pieces together with a separator between
        // them (not after the last one). `\"` prints a literal quote mark.
        outln!(
            "  \"{}\" -> {} tokens: [{}]",
            example,
            token_ids.len(),
            token_strs.join("|")
        );
    }

    outln!("\n{}\n", "=".repeat(60));

    report
}


// --- 7. Sizing and sampling ---

/// Sample the corpus text.
///
/// For large vocabularies, you may want to choose to use subset of the corpus
/// to learn patterns faster.
///
/// For typical English prose, a 200KB sample such as the first 200KB of the
/// text is usually sufficient to learn common patterns.
///
/// For more-sophisticated text, such as complex terminology, a larger sample
/// may be necessary to capture relevant patterns.
///
/// The text is truncated only when both conditions hold: it is longer than
/// `text_len_max` bytes, and more than `num_merges_max` merges are wanted.
/// Otherwise the full text is returned. Not currently called by `train`.
///
/// # Arguments
///
/// * `text` - the full corpus.
/// * `text_len_max` - the maximum number of bytes to keep when sampling.
/// * `num_merges` - the number of merges requested.
/// * `num_merges_max` - the number of merges above which sampling kicks in.
///
/// # Returns
///
/// A prefix of `text`, cut at a UTF-8 character boundary.
///
pub fn text_sample(text: &str, text_len_max: usize, num_merges: usize, num_merges_max: usize) -> &str {
    if text.len() > text_len_max && num_merges > num_merges_max  {
        // `floor_char_boundary` rounds down to a valid boundary, so the slice
        // never splits a multi-byte character (which would panic).
        &text[..text.floor_char_boundary(text_len_max)]
    } else {
        text
    }
}

/// Calculate a good chunk size suitable for parallelization.
///
/// The chunk size is based on the number of tokens, the chunk size min,
/// and the size of the thread pool, with an error guard for a value of 0
/// threads.
///
/// Splitting the tokens evenly across threads gives about one chunk per
/// thread, which minimizes scheduling overhead; the minimum prevents tiny
/// chunks on small inputs. The result is always at least `chunk_size_min`
/// (and so non-zero if the minimum is, which `par_chunks` requires).
///
/// Only the *count* of tokens matters, so this takes a number rather than the
/// list itself.
///
/// # Arguments
///
/// * `token_count` - how many tokens are to be split.
/// * `chunk_size_min` - the smallest acceptable chunk size.
pub fn chunk_size(token_count: usize, chunk_size_min: usize) -> usize {
    // Worked example: 1,000,000 tokens on 8 threads, minimum 50,000.
    //   tokens / threads = 125,000, which beats the minimum, so chunks are 125,000
    //   and `par_chunks` yields exactly 8 chunks, one per thread.
    // With only 100,000 tokens: 100,000 / 8 = 12,500 < 50,000, so the minimum
    // wins, giving 2 chunks; using all 8 threads would cost more than it saves.
    // `.max(1)` guards the division against a (theoretical) zero thread count.
    chunk_size_min.max(token_count / rayon::current_num_threads().max(1))
}

// Tests for the pure functions. They call the functions directly: no
// Tokenizer is needed, only the tables the functions take as arguments.
// Tests of the Tokenizer itself (training, invariants, round trips, files) are
// in `tokenizer.rs`.
#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Free functions
    // ------------------------------------------------------------------

    /// Each byte becomes one token whose ID is the byte value.
    #[test]
    fn test_from_text_into_token_ids() {
        assert_eq!(from_text_into_token_ids("hi"), vec![0x68, 0x69]);
        assert!(from_text_into_token_ids("").is_empty());
    }

    /// A multi-byte UTF-8 character yields one token per byte.
    #[test]
    fn test_from_text_into_token_ids_multibyte() {
        assert_eq!(from_text_into_token_ids("é"), vec![0xc3, 0xa9]);
        assert_eq!(from_text_into_token_ids("日").len(), 3);
    }

    /// Short text is returned whole, whatever the merge count.
    #[test]
    fn test_text_sample_short_text_unchanged() {
        assert_eq!(text_sample("hello", 100, 5000, 2000), "hello");
    }

    /// Long text with few merges is returned whole (sampling needs both conditions).
    #[test]
    fn test_text_sample_few_merges_unchanged() {
        let text = "a".repeat(1000);
        assert_eq!(text_sample(&text, 100, 10, 2000).len(), 1000);
    }

    /// Long text with many merges is cut to the byte limit.
    #[test]
    fn test_text_sample_truncates() {
        let text = "a".repeat(1000);
        assert_eq!(text_sample(&text, 100, 5000, 2000).len(), 100);
    }

    /// Truncation never splits a multi-byte character (that would panic).
    #[test]
    fn test_text_sample_respects_char_boundary() {
        let text = "é".repeat(10); // 20 bytes, each char 2 bytes
        // Byte 5 is in the middle of a character, so the cut moves down to 4.
        assert_eq!(text_sample(&text, 5, 5000, 2000), "éé");
    }

    /// Small inputs get the minimum chunk size.
    #[test]
    fn test_chunk_size_minimum_wins_for_small_input() {
        assert_eq!(chunk_size(10, 50_000), 50_000);
    }

    /// Large inputs are divided across the thread pool, and never below the minimum.
    #[test]
    fn test_chunk_size_divides_by_threads() {
        let expect = (400_000 / rayon::current_num_threads().max(1)).max(1);
        assert_eq!(chunk_size(400_000, 1), expect);
        assert!(chunk_size(400_000, 300_000) >= 300_000);
    }

    // ------------------------------------------------------------------
    // merge_pair_in_place
    // ------------------------------------------------------------------

    /// Matches are replaced and the list shrinks.
    #[test]
    fn test_merge_pair_in_place_basic() {
        let mut token_ids = vec![1, 2, 3, 1, 2];
        merge_pair_in_place(&mut token_ids, (1, 2), 9);
        assert_eq!(token_ids, vec![9, 3, 9]);
    }

    /// Matches are found left to right and never overlap: [1,1,1] -> [9,1].
    #[test]
    fn test_merge_pair_in_place_no_overlap() {
        let mut token_ids = vec![1, 1, 1];
        merge_pair_in_place(&mut token_ids, (1, 1), 9);
        assert_eq!(token_ids, vec![9, 1]);

        let mut token_ids = vec![1, 1, 1, 1];
        merge_pair_in_place(&mut token_ids, (1, 1), 9);
        assert_eq!(token_ids, vec![9, 9]);
    }

    /// With no match, the list is unchanged.
    #[test]
    fn test_merge_pair_in_place_no_match() {
        let mut token_ids = vec![1, 2, 3];
        merge_pair_in_place(&mut token_ids, (3, 1), 9);
        assert_eq!(token_ids, vec![1, 2, 3]);
    }

    /// Empty and single-token lists are handled without panicking.
    #[test]
    fn test_merge_pair_in_place_tiny_inputs() {
        let mut empty: Vec<TokenId> = vec![];
        merge_pair_in_place(&mut empty, (1, 2), 9);
        assert!(empty.is_empty());

        let mut one = vec![1];
        merge_pair_in_place(&mut one, (1, 2), 9);
        assert_eq!(one, vec![1]);
    }

    /// Order matters: (1,2) is only a pair when 1 comes first.
    #[test]
    fn test_merge_pair_in_place_respects_order() {
        let mut token_ids = vec![2, 1];
        merge_pair_in_place(&mut token_ids, (1, 2), 9);
        assert_eq!(token_ids, vec![2, 1]);
    }

    // ------------------------------------------------------------------
    // Pure functions
    //
    // These call the free functions directly: no Tokenizer is needed, only
    // the tables they take as arguments.
    // ------------------------------------------------------------------

    #[test]
    fn test_base_tokens() {
        let tokens = base_tokens();
        let actual = tokens.len();
        let expect = 256;
        assert_eq!(actual, expect);
        for byte in 0..=255u8 {
            let actual = &tokens[byte as usize];
            let expect = &vec![byte];
            assert_eq!(actual, expect);
        }
    }

    #[test]
    fn test_new_token_from_pair() {
        let tokens = base_tokens();
        let pair = (0x68, 0x69); // "h", "i"
        let actual = new_token_from_pair(&tokens, pair);
        let expect = b"hi".to_vec();
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_build_tokens_from_merges_ok() {
        let merges = vec![(97, 97), (256, 256)];
        let actual = build_tokens_from_merges(&merges).unwrap();
        assert_eq!(actual.len(), 258);
        assert_eq!(actual[256], b"aa".to_vec());
        assert_eq!(actual[257], b"aaaa".to_vec());
    }

    #[test]
    fn test_build_tokens_from_merges_names_the_bad_rule() {
        // Rule 1 uses ID 300, which does not exist yet.
        let merges = vec![(97, 97), (300, 1)];
        let actual = build_tokens_from_merges(&merges).unwrap_err();
        assert!(actual.contains("invalid merge rule 1"), "message: {}", actual);
    }

    #[test]
    fn test_count_token_id_pairs_small() {
        let token_ids = vec![1, 1, 2];
        let actual = count_token_id_pairs(&token_ids);
        let mut expect = TokenIdPairToCountMap::new();
        expect.insert((1, 1), 1);
        expect.insert((1, 2), 1);
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_count_token_id_pairs_needs_two_tokens() {
        let empty: Vec<TokenId> = vec![];
        assert!(count_token_id_pairs(&empty).is_empty());
        assert!(count_token_id_pairs(&[7]).is_empty());
    }

    #[test]
    fn test_count_token_id_pairs_matches_naive_count_across_chunks() {
        // More than 50,000 tokens, so the parallel path can use several chunks,
        // and the pairs that straddle a chunk boundary must still be counted.
        let token_ids: Vec<TokenId> = (0..130_000u64)
            .map(|i| ((i * i + i / 3) % 7) as TokenId)
            .collect();
        let mut expect = TokenIdPairToCountMap::new();
        for window in token_ids.windows(2) {
            *expect.entry((window[0], window[1])).or_insert(0) += 1;
        }
        let actual = count_token_id_pairs(&token_ids);
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_select_best_pair_highest_count_wins() {
        let tokens = base_tokens();
        let mut counts = TokenIdPairToCountMap::new();
        counts.insert((97, 97), 4);
        counts.insert((97, 98), 2);
        let actual = select_best_pair(counts, &tokens);
        let expect = Some(((97, 97), 4));
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_select_best_pair_tie_goes_to_smaller_bytes() {
        let tokens = base_tokens();
        let mut counts = TokenIdPairToCountMap::new();
        counts.insert((97, 98), 3); // "ab"
        counts.insert((97, 97), 3); // "aa" sorts first
        counts.insert((98, 97), 3); // "ba"
        let actual = select_best_pair(counts, &tokens);
        let expect = Some(((97, 97), 3));
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_select_best_pair_equal_bytes_goes_to_smaller_ids() {
        // Tokens 256 and 257 both stand for "aa", so (256, 98) and (257, 98)
        // have identical bytes; the smaller IDs must win.
        let mut tokens = base_tokens();
        tokens.push(b"aa".to_vec());
        tokens.push(b"aa".to_vec());
        let mut counts = TokenIdPairToCountMap::new();
        counts.insert((257, 98), 2);
        counts.insert((256, 98), 2);
        let actual = select_best_pair(counts, &tokens);
        let expect = Some(((256, 98), 2));
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_select_best_pair_empty_is_none() {
        let tokens = base_tokens();
        let actual = select_best_pair(TokenIdPairToCountMap::new(), &tokens);
        let expect = None;
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_apply_merges() {
        let merges = vec![(97, 97)];
        let mut actual = vec![97, 97, 97];
        apply_merges(&merges, &mut actual);
        let expect = vec![256, 97];
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_apply_merges_without_rules_changes_nothing() {
        let mut actual = vec![1, 2, 3];
        apply_merges(&[], &mut actual);
        let expect = vec![1, 2, 3];
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_encode_sequential_replays_rules_in_order() {
        // (104, 105) = "hi" -> 256, then (256, 33) = "hi!" -> 257.
        let merges = vec![(104, 105), (256, 33)];
        let actual = encode_sequential(&merges, "hi!");
        let expect = vec![257];
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_split_text_into_chunks_covers_text_exactly() {
        let text = "abcdefghij";
        let actual = split_text_into_chunks(text, 4);
        let expect = vec!["abcd", "efgh", "ij"];
        assert_eq!(actual, expect);
        assert_eq!(actual.concat(), text);
    }

    #[test]
    fn test_split_text_into_chunks_never_splits_a_character() {
        // "é" is two bytes, so a 3-byte cut would land inside the second "é".
        let text = "aéé";
        let actual = split_text_into_chunks(text, 3);
        assert_eq!(actual.concat(), text);
        for chunk in &actual {
            assert!(!chunk.is_empty());
        }
    }

    #[test]
    fn test_split_text_into_chunks_empty_text() {
        let actual = split_text_into_chunks("", 10);
        let expect: Vec<&str> = vec![];
        assert_eq!(actual, expect);
    }

    #[test]
    #[should_panic(expected = "at least 1 byte")]
    fn test_split_text_into_chunks_zero_size_panics() {
        let _ = split_text_into_chunks("abc", 0);
    }

    #[test]
    fn test_merges_json_round_trip() {
        let merges = vec![(97, 97), (256, 256)];
        let json = merges_into_json(&merges).unwrap();
        let actual = merges_from_json(&json).unwrap();
        let expect = merges;
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_merges_into_json_has_version_and_no_tokens_table() {
        let json = merges_into_json(&[(97, 97)]).unwrap();
        assert!(json.contains("\"format_version\": 1"), "json: {}", json);
        assert!(json.contains("\"merges\""), "json: {}", json);
        assert!(!json.contains("\"tokens\""), "json: {}", json);
    }

    #[test]
    fn test_merges_from_json_rejects_wrong_version() {
        let json = r#"{"format_version": 99, "merges": []}"#;
        let actual = merges_from_json(json).unwrap_err().to_string();
        assert!(actual.contains("unsupported tokenizer format_version 99"), "message: {}", actual);
    }

    #[test]
    fn test_merges_from_json_rejects_old_layout_and_garbage() {
        // No format_version: the pre-redesign layout.
        assert!(merges_from_json(r#"{"merges": []}"#).is_err());
        assert!(merges_from_json("not json").is_err());
    }

    #[test]
    fn test_learned_token_samples_limits_to_first_n_learned_tokens() {
        let mut tokens = base_tokens();
        tokens.push(b"aa".to_vec()); // 256
        tokens.push(b"bb".to_vec()); // 257
        tokens.push(b"cc".to_vec()); // 258
        let actual = learned_token_samples(&tokens, 2);
        let expect = vec![(256, "aa".to_string()), (257, "bb".to_string())];
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_learned_token_samples_skips_partial_characters_and_long_tokens() {
        let mut tokens = base_tokens();
        tokens.push(vec![0xC3]); // 256: half of "é", shows as U+FFFD
        tokens.push(vec![b'x'; 21]); // 257: longer than 20 bytes
        tokens.push(b"ok".to_vec()); // 258
        let actual = learned_token_samples(&tokens, 30);
        let expect = vec![(258, "ok".to_string())];
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_compression_ratio() {
        let actual = compression_ratio(100, 40);
        let expect = 2.5;
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_token_strings_marks_unknown_ids() {
        let mut tokens = base_tokens();
        tokens.push(b"he".to_vec()); // 256
        let actual = token_strings(&tokens, &[256, 98, 999]);
        let expect = vec!["he".to_string(), "b".to_string(), "?".to_string()];
        assert_eq!(actual, expect);
    }

    // ------------------------------------------------------------------
    // train, encode, decode and vocabulary_report: the tokenizer operations
    // ------------------------------------------------------------------

    /// A tokenizer trained on `text` up to `vocab_size`.
    fn trained_on(text: &str, vocab_size: usize) -> Tokenizer {
        let mut tokenizer = Tokenizer::default();
        train(&mut tokenizer, text, vocab_size);
        tokenizer
    }

    #[test]
    fn test_train_first_rule() {
        let tokenizer = trained_on("aaaa", 257);
        assert_eq!(tokenizer.merges, vec![(97, 97)]);
        assert_eq!(tokenizer.tokens.len(), 257);
        assert_eq!(tokenizer.tokens[256], b"aa".to_vec());
    }

    #[test]
    fn test_train_later_rules_build_on_earlier_ones() {
        let tokenizer = trained_on("aaaaaaaa", 258);
        // "aa" = 256, then (256, 256) = "aaaa" = 257.
        assert_eq!(tokenizer.merges, vec![(97, 97), (256, 256)]);
        assert_eq!(tokenizer.tokens[257], b"aaaa".to_vec());
    }

    #[test]
    fn test_train_target_at_or_below_current_size_does_nothing() {
        let mut tokenizer = Tokenizer::default();
        train(&mut tokenizer, "aaaa", 256);
        train(&mut tokenizer, "aaaa", 10);
        assert!(tokenizer.merges.is_empty());
        assert_eq!(tokenizer.tokens.len(), 256);
    }

    #[test]
    fn test_train_stops_early_when_no_pairs_remain() {
        // "ab" has one pair; after merging it the corpus is a single token.
        let tokenizer = trained_on("ab", 300);
        assert_eq!(tokenizer.merges, vec![(97, 98)]);
        assert_eq!(tokenizer.tokens.len(), 257);
    }

    #[test]
    fn test_train_resumes_where_it_stopped() {
        let text = "the cat and the hat and the bat ".repeat(20);
        let straight = trained_on(&text, 280);

        let mut resumed = trained_on(&text, 265);
        train(&mut resumed, &text, 280);

        assert_eq!(straight.merges, resumed.merges);
        assert_eq!(straight.tokens, resumed.tokens);
    }

    #[test]
    fn test_train_keeps_tables_consistent() {
        let tokenizer = trained_on("hello world hello rust hello world", 280);
        // One table entry per token: 256 + one per rule.
        assert_eq!(tokenizer.tokens.len(), 256 + tokenizer.merges.len());
        // The table can be rebuilt from the rules alone.
        assert_eq!(build_tokens_from_merges(&tokenizer.merges).unwrap(), tokenizer.tokens);
    }

    #[test]
    fn test_encode_large_text_takes_the_parallel_path_and_round_trips() {
        // "hi!" repeated is 300,000 bytes (over the 200,000 threshold), with the
        // rules "hi" -> 256 and "hi!" -> 257.
        let tokenizer = Tokenizer::from_merges(vec![(104, 105), (256, 33)]).unwrap();
        let text = "hi!".repeat(100_000);
        let ids = encode(&tokenizer, &text);
        let actual = decode(&tokenizer, &ids);
        let expect = text;
        assert_eq!(actual, expect);
        // And it really did compress: far fewer tokens than bytes.
        assert!(ids.len() < expect.len() / 2);
    }

    #[test]
    fn test_encode_small_text_matches_sequential() {
        let tokenizer = Tokenizer::from_merges(vec![(104, 105), (256, 33)]).unwrap();
        let actual = encode(&tokenizer, "hi!");
        let expect = encode_sequential(&tokenizer.merges, "hi!");
        assert_eq!(actual, expect);
        assert_eq!(actual, vec![257]);
    }

    #[test]
    fn test_decode_skips_unknown_ids() {
        let tokenizer = Tokenizer::default();
        let actual = decode(&tokenizer, &[104, 999, 105]);
        let expect = "hi";
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_decode_replaces_partial_character() {
        // 0xC3 alone is the first half of "é": invalid UTF-8 by itself.
        let tokenizer = Tokenizer::default();
        let actual = decode(&tokenizer, &[0xC3]);
        let expect = "\u{FFFD}";
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_vocabulary_report_for_untrained_tokenizer() {
        let tokenizer = Tokenizer::default();
        let actual = vocabulary_report(&tokenizer, "");
        assert!(actual.contains("=== Vocabulary Analysis ==="), "report: {}", actual);
        assert!(actual.contains("Base tokens (bytes): 256"));
        assert!(actual.contains("Learned merges: 0"));
        assert!(actual.contains("Total vocabulary: 256"));
        // No sample text, so no compression section.
        assert!(!actual.contains("Compression Analysis"));
        // The fixed example phrases are always shown. 19 bytes, 19 base tokens.
        assert!(actual.contains("\"To be, or not to be\" -> 19 tokens"));
    }

    #[test]
    fn test_vocabulary_report_compression_section() {
        // With no merges every byte is its own token: ratio 1.00x.
        let tokenizer = Tokenizer::default();
        let actual = vocabulary_report(&tokenizer, "hello");
        assert!(actual.contains("Sample size: 5 bytes"), "report: {}", actual);
        assert!(actual.contains("Token count: 5 tokens"));
        assert!(actual.contains("Compression ratio: 1.00x"));
    }

    #[test]
    fn test_vocabulary_report_lists_learned_tokens() {
        let tokenizer = trained_on("aaaa", 257);
        let actual = vocabulary_report(&tokenizer, "aaaa");
        assert!(actual.contains("Learned merges: 1"), "report: {}", actual);
        assert!(actual.contains("[256] \"aa\""));
        // "aaaa" encodes to 2 tokens: ratio 4 / 2 = 2.00x.
        assert!(actual.contains("Compression ratio: 2.00x"));
    }
}
