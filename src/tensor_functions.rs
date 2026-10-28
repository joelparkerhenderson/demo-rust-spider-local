//! Pure functions for tensors: the tensor operations
//!
//! Project convention (see `spec/index.md`): prefer pure functions, not `self`
//! methods, and make functions public by default. Each function here takes
//! everything it needs as arguments and returns its result, with no hidden
//! state. The tensor operations ([`matmul`], [`softmax`], [`add`], [`transpose`]
//! and the rest) are free functions that take [`Tensor`]s and return a NEW
//! `Tensor`: `matmul(&a, &b)`, not `a.matmul(&b)`. The `Tensor` type in
//! `tensor.rs` keeps only what needs its own fields: construction (`new`,
//! `zeros`, `arange`), `reshape`, and the scalar operations.
//!
//! Below the operations are lower-level helpers that work on plain slices and
//! shapes (`&[f32]`, `&[usize]`), so they can be read and tested on their own.
//!
//! ## What is in this file
//!
//! 1. Shapes and strides: [`compute_strides`], [`normalize_axis`]
//! 2. Broadcasting: [`broadcast_shapes`], [`broadcast_strides`], [`broadcast_zip`]
//! 3. Matrix multiply: [`matmul`] (check shapes, choose, compute), with the
//!    helpers [`matmul_2d`], [`matmul_batched`], [`matmul_sequential`],
//!    [`matmul_sequential_into`], [`matmul_parallel_blocked`], [`matmul_inner_simd`]
//! 4. Softmax: [`softmax`], with the helper [`softmax_slice`]
//! 5. The other tensor operations: [`transpose`], [`masked_fill`], [`mean`],
//!    [`var`], [`concat()`], and the element-wise [`add`], [`sub`], [`mul`], [`div`]
//!    (which broadcast, using [`broadcast_zip`])
//!
//! ## The mental model, in short
//!
//! A tensor is a flat list of numbers plus a shape. For shape `[2, 3, 4]` the
//! row-major strides are `[12, 4, 1]`, and element `[i, j, k]` lives at flat
//! index `i*12 + j*4 + k`. See the module docs of `tensor.rs` for the full
//! tutorial. Each operation below has its own worked examples.

use crate::tensor::Tensor;
use rayon::prelude::*;

/// Compute strides from shape (row-major layout)
///
/// For shape `[d0, d1, d2]`, strides are `[d1*d2, d2, 1]`
///
/// # Arguments
///
/// * `shape` - Tensor dimensions
///
/// # Returns
///
/// Vector of stride values for each dimension
///
/// # Worked example
///
/// ```text
/// shape = [2, 3, 4]
///
/// start:  strides = [1, 1, 1]            (the last one stays 1)
///
/// i = 1:  strides[1] = strides[2] * shape[2] = 1 * 4  = 4
/// i = 0:  strides[0] = strides[1] * shape[1] = 4 * 3  = 12
///
/// result: strides = [12, 4, 1]
/// ```
///
/// Reading it in words: to move one step along a dimension, skip over
/// everything that lives inside one step of the dimension after it.
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    // Start with all ones. The LAST stride is always 1, because the last
    // dimension is contiguous in memory.
    let mut strides = vec![1; shape.len()];

    // Fill in the rest, right to left. The loop covers indices
    // `len-2` down to `0`.
    //
    // `saturating_sub(1)` is a guard for the empty shape (a scalar):
    // `0 - 1` would underflow a `usize` and panic, whereas the saturating
    // version stops at 0, making the range `0..0` (empty), so the loop
    // simply does not run and the empty `strides` is returned.
    for i in (0..shape.len().saturating_sub(1)).rev() {
        // Stride of dimension i = stride of the next dimension times the
        // size of the next dimension.
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Convert a possibly negative axis into a normal index, with a range check
///
/// A pure function: it looks only at the shape it is given, so it can be used
/// (and tested) without building a tensor.
///
/// # Arguments
///
/// * `shape` - The shape of the tensor the axis refers to
/// * `axis` - The axis, where negative values count from the end
/// * `op` - Operation name, used only in the error message
///
/// Python-style: `-1` is the last axis, `-2` the one before it. Adding the
/// number of dimensions turns a negative axis into a normal index: for a
/// 3-D tensor, `-1 + 3 = 2`.
///
/// # Panics
///
/// Panics if `axis` is outside `-ndim..ndim`. Checking here gives one clear
/// message instead of a confusing "index out of bounds" later.
pub fn normalize_axis(shape: &[usize], axis: isize, op: &str) -> usize {
    let ndim = shape.len() as isize;
    assert!(
        axis >= -ndim && axis < ndim,
        "{}: axis {} is out of range for shape {:?}",
        op,
        axis,
        shape
    );
    if axis < 0 {
        (ndim + axis) as usize
    } else {
        axis as usize
    }
}

/// Compute the shape that results from broadcasting two shapes together
///
/// Returns `None` if the shapes cannot be broadcast.
///
/// # The rule (same as NumPy and PyTorch)
///
/// 1. Line the shapes up from the RIGHT (last dimensions together). The
///    shorter shape is treated as if it had leading 1s.
/// 2. For each pair of sizes, they are compatible if they are equal or
///    one of them is 1. The result takes the larger one.
///
/// ```text
/// a:      [2, 3, 4]          a:      [2, 3]
/// b:         [3, 1]          b:      [2, 1]
/// result: [2, 3, 4]          result: [2, 3]
///
/// a:      [2, 3]             a:      [3]
/// b:      [4, 3]  -> None    b:      [2, 3]  -> [2, 3]
/// ```
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let ndim = a.len().max(b.len());
    let mut out = vec![0; ndim];
    // `k` counts from the right: k = 0 is the last dimension.
    for k in 0..ndim {
        let da = if k < a.len() { a[a.len() - 1 - k] } else { 1 };
        let db = if k < b.len() { b[b.len() - 1 - k] } else { 1 };
        out[ndim - 1 - k] = if da == db || db == 1 {
            da
        } else if da == 1 {
            db
        } else {
            return None;
        };
    }
    Some(out)
}

/// Strides to read a tensor as if it had the broadcast shape `out_shape`
///
/// # Arguments
///
/// * `shape` - The tensor's own shape
/// * `strides` - The tensor's own (row-major) strides
/// * `out_shape` - The broadcast shape to read it as
///
/// A dimension that is missing, or has size 1, gets stride 0. A stride of
/// 0 means "do not move": the same number is read again for every step
/// along that dimension. That is how a small tensor is "repeated" without
/// storing any copies.
///
/// ```text
/// bias  shape [3],   strides [1]
/// out   shape [2, 3]
/// bias read as shape [2, 3] uses strides [0, 1]
///                                         ^ row step: stay on the same bias
/// ```
pub fn broadcast_strides(shape: &[usize], strides: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let ndim = out_shape.len();
    // How many leading dimensions the tensor is missing.
    let offset = ndim - shape.len();
    (0..ndim)
        .map(|k| {
            if k < offset || shape[k - offset] == 1 {
                0
            } else {
                strides[k - offset]
            }
        })
        .collect()
}

/// Apply `f` to every pair of numbers from two (broadcast) arrays
///
/// This is the one engine behind `add`, `sub`, `mul` and `div`. `f(x, y)`
/// always receives `a`'s number first, so non-commutative operations
/// such as subtraction keep their order even when `a` is the smaller
/// array.
///
/// # Steps
///
/// 1. Same shape: pair the numbers directly (the fast path).
/// 2. Otherwise compute the broadcast output shape (panic if impossible).
/// 3. For every output position, decode its coordinates, then find the
///    matching position in each input using that input's broadcast strides.
///
/// ```text
/// matrix [2, 3] + bias [3]  ->  out shape [2, 3], out strides [3, 1]
/// matrix strides [3, 1]     bias strides [0, 1]
///
/// output index i = 4:
///   coordinates: 4 / 3 = 1 (row), remainder 1 (column)  -> [1, 1]
///   matrix index = 1*3 + 1*1 = 4
///   bias   index = 1*0 + 1*1 = 1
/// ```
///
/// # Arguments
///
/// * `a_data` - the first array's numbers, in row-major order
/// * `a_shape` - the first array's shape
/// * `b_data` - the second array's numbers, in row-major order
/// * `b_shape` - the second array's shape
/// * `op` - the operation's name, used only in the error message
/// * `f` - the operation to apply to each pair of numbers
///
/// # Returns
///
/// `(data, shape)`: the result's numbers in row-major order, and the broadcast
/// shape.
///
/// # Panics
///
/// Panics if the two shapes cannot be broadcast together.
pub fn broadcast_zip(
    a_data: &[f32],
    a_shape: &[usize],
    b_data: &[f32],
    b_shape: &[usize],
    op: &str,
    f: impl Fn(f32, f32) -> f32 + Sync,
) -> (Vec<f32>, Vec<usize>) {
    // === EXACT MATCH: Same shape ===
    //
    // `par_iter` splits the slice across threads, `zip` pairs element i
    // of each array, `map` applies `f`, and `collect` gathers the
    // results into a Vec in the original order.
    if a_shape == b_shape {
        let result = a_data
            .par_iter()
            .zip(b_data)
            .map(|(&x, &y)| f(x, y))
            .collect();
        return (result, a_shape.to_vec());
    }

    // === BROADCAST ===
    let out_shape = match broadcast_shapes(a_shape, b_shape) {
        Some(shape) => shape,
        None => panic!(
            "Cannot broadcast shapes {:?} and {:?} for {}",
            a_shape, b_shape, op
        ),
    };

    // Row-major strides of the output and of each input (read as the output).
    let out_strides = compute_strides(&out_shape);
    let a_strides = broadcast_strides(a_shape, &compute_strides(a_shape), &out_shape);
    let b_strides = broadcast_strides(b_shape, &compute_strides(b_shape), &out_shape);
    let size: usize = out_shape.iter().product();

    let result: Vec<f32> = (0..size)
        .into_par_iter()
        .map(|i| {
            // Peel off one coordinate per dimension: divide by the output
            // stride to get the coordinate, keep the remainder for the
            // next dimension. Accumulate each input's flat index on the way.
            let mut remaining = i;
            let mut a_idx = 0;
            let mut b_idx = 0;
            for k in 0..out_shape.len() {
                let coord = remaining / out_strides[k];
                remaining %= out_strides[k];
                a_idx += coord * a_strides[k];
                b_idx += coord * b_strides[k];
            }
            f(a_data[a_idx], b_data[b_idx])
        })
        .collect();

    (result, out_shape)
}

/// SIMD-optimized inner loop for matrix multiplication
/// Computes: `result[j] += a_val * b[j]` for all `j`
/// Structured for auto-vectorization on ARM NEON (Apple Silicon)
///
/// # What this does
///
/// It adds "one number of A, times one whole row-slice of B" into a
/// slice of the output. Written as a plain loop, the compiler can turn
/// it into SIMD instructions that multiply and add several `f32` values
/// per CPU instruction (4 at a time with ARM NEON, 8 with AVX).
///
/// ```text
/// a_val = 2
/// b      = [ 1  10  100 ]
/// result = [ 5   5    5 ]   before
///
/// result += a_val * b  ->  [ 5+2  5+20  5+200 ]  =  [ 7  25  205 ]
/// ```
///
/// Why it exists as its own function: the output row of a matrix
/// product is a sum of scaled rows of B (see [`matmul`]). This
/// is that step, isolated so the optimizer can focus on it.
///
/// `#[inline(always)]` asks the compiler to paste the body into the
/// caller. The function is tiny and is called in the hottest loop, so
/// the cost of a real function call would be significant.
#[inline(always)]
pub fn matmul_inner_simd(a_val: f32, b: &[f32], result: &mut [f32]) {
    // Simple loop that LLVM can auto-vectorize
    // On Apple Silicon this will use ARM NEON SIMD instructions
    //
    // `zip` pairs each output slot with the matching element of `b` and
    // stops at the shorter slice (the caller passes equal lengths).
    // `iter_mut` hands out `&mut f32`, so `*r += ...` updates in place.
    // There is no indexing, so there are no bounds checks inside the
    // loop; that also helps the compiler vectorize.
    for (r, &b_val) in result.iter_mut().zip(b.iter()) {
        *r += a_val * b_val;
    }
}

/// Parallel cache-blocked matrix multiplication
///
/// This optimization provides significant speedup for large matrices through:
///
/// 1. **Cache blocking**: Processes data in 8×8 blocks that fit in L1 cache
/// 2. **Parallel processing**: Distributes row blocks across CPU cores via Rayon
/// 3. **Memory locality**: Inner loops access memory sequentially
///
/// The 8×8 block size (256 bytes per block) balances cache efficiency with
/// parallelism opportunities. Smaller blocks would improve cache hit rate but
/// reduce available parallelism; larger blocks do the opposite.
///
/// # Performance
///
/// Typically 2-4x faster than naive implementation on multi-core CPUs,
/// with speedup increasing for larger matrices.
///
/// # Arguments
///
/// * `a` - Left-hand matrix, `m * k` numbers in row-major order
/// * `b` - Right-hand matrix, `k * n` numbers in row-major order
/// * `m` - Rows in `a`
/// * `n` - Columns in `b`
/// * `k` - Inner dimension (columns of `a`, rows of `b`)
///
/// # Returns
///
/// The `m * n` result numbers in row-major order, i.e. a matrix of shape `[m, n]`.
///
/// # Panics
///
/// Panics (index out of bounds) if `a` or `b` is shorter than the sizes say.
///
/// # How it works, step by step
///
/// The math is the same as the simple triple loop. What changes is the
/// ORDER of the work, to be kind to the CPU cache and to use all cores.
///
/// 1. **Split the output into bands of 8 rows.** Each band is given to
///    one worker thread. Bands do not overlap, so no locks are needed.
/// 2. **Inside a band, visit the work in 8-wide tiles** (j = column
///    tile, k = inner-dimension tile). The data for one tile stays in
///    the fast L1 cache while it is reused.
/// 3. **Use the "row of B" form of the product.** Instead of one dot
///    product per output cell, take each number `A[i][k]` and add
///    `A[i][k] * (row k of B)` into output row `i`. Both the B row and the
///    output row are read in order (sequential memory), which is fast.
///
/// ```text
/// output row i  =  sum over k of   A[i][k]  *  (row k of B)
///
///   A[i] = [ 2  3 ]    B = [ 1 0 ]      out[i] = 2 * [1 0] + 3 * [0 1]
///                          [ 0 1 ]             = [2 0] + [0 3] = [2 3]
/// ```
///
/// The result is identical to the simple loop up to rounding: floating
/// point addition is not associative, so summing in a different order
/// can change the last digit.
pub fn matmul_parallel_blocked(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    // Block size for cache optimization
    // 8×8 blocks = 256 bytes (fits well in L1 cache: typically 32-64KB)
    //
    // 8 * 8 = 64 numbers, 4 bytes each = 256 bytes.
    const BLOCK_SIZE: usize = 8;

    let mut result = vec![0.0; m * n];

    // Parallelize over output row blocks
    // Each thread processes BLOCK_SIZE rows independently
    //
    // One chunk is BLOCK_SIZE rows of `n` numbers. The final chunk may
    // be shorter when `m` is not a multiple of 8; `i_end` below handles it.
    // `block_i` is the band number (0, 1, 2, ...).
    result
        .par_chunks_mut(BLOCK_SIZE * n)
        .enumerate()
        .for_each(|(block_i, result_block)| {
            // The first and one-past-last rows of the output covered by
            // this band. `.min(m)` clips the last, partial band.
            let i_start = block_i * BLOCK_SIZE;
            let i_end = (i_start + BLOCK_SIZE).min(m);

            // Iterate over column blocks
            for j_start in (0..n).step_by(BLOCK_SIZE) {
                let j_end = (j_start + BLOCK_SIZE).min(n);

                // Iterate over inner dimension blocks
                for k_start in (0..k).step_by(BLOCK_SIZE) {
                    let k_end = (k_start + BLOCK_SIZE).min(k);

                    // Compute this block (cache-friendly inner loops)
                    for i in i_start..i_end {
                        // `result_block` starts at this band's first row,
                        // so a row's offset within it is relative:
                        // (i - i_start) rows, each `n` numbers wide.
                        let row_offset = (i - i_start) * n;
                        for k_idx in k_start..k_end {
                            // The one number of A we are scaling by.
                            let a_val = a[i * k + k_idx];

                            // SIMD-optimized innermost loop
                            //
                            // result_row[j_start..j_end] += a_val * B_row_k[j_start..j_end]
                            // Both slices are `j_end - j_start` long.
                            matmul_inner_simd(
                                a_val,
                                &b[k_idx * n + j_start..k_idx * n + j_end],
                                &mut result_block[row_offset + j_start..row_offset + j_end],
                            );
                        }
                    }
                }
            }
        });

    result
}

/// Multiply two matrices with the textbook three nested loops, into a given buffer
///
/// `C[i][j] = sum over l of A[i][l] * B[l][j]`, a direct transcription of the
/// definition. Writing into `out` (instead of returning a new `Vec`) lets
/// [`matmul_batched`] fill one slice of its result per matrix pair.
///
/// # Arguments
///
/// * `a` - Left-hand matrix, `m * k` numbers in row-major order
/// * `b` - Right-hand matrix, `k * n` numbers in row-major order
/// * `out` - Where to write the `m * n` result numbers (row-major)
/// * `m` - Rows in `a`
/// * `n` - Columns in `b`
/// * `k` - Inner dimension (columns of `a`, rows of `b`)
pub fn matmul_sequential_into(a: &[f32], b: &[f32], out: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                // Flat indices (row-major):
                //   A[i][l] -> a[i * k + l]   (A has k columns)
                //   B[l][j] -> b[l * n + j]   (B has n columns)
                sum += a[i * k + l] * b[l * n + j];
            }
            // C[i][j] -> out[i * n + j]   (C has n columns)
            out[i * n + j] = sum;
        }
    }
}

/// Multiply two matrices with the textbook three nested loops (no threads)
///
/// [`matmul_2d`] uses it for small products, where starting threads would cost
/// more than the work.
///
/// # Arguments
///
/// * `a` - Left-hand matrix, `m * k` numbers in row-major order
/// * `b` - Right-hand matrix, `k * n` numbers in row-major order
/// * `m` - Rows in `a`
/// * `n` - Columns in `b`
/// * `k` - Inner dimension (columns of `a`, rows of `b`)
///
/// # Returns
///
/// The `m * n` result numbers in row-major order.
pub fn matmul_sequential(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut result = vec![0.0; m * n];
    matmul_sequential_into(a, b, &mut result, m, n, k);
    result
}

/// Multiply two matrices, choosing the sequential or the parallel algorithm
///
/// Use parallel version for larger matrices (work threshold: 1000 operations).
/// This threshold balances parallel overhead against performance gains.
///
/// `m * n * k` counts the multiply-adds. Starting threads costs a few
/// microseconds, so for tiny products it is faster to stay on one thread:
/// [`matmul_sequential`]. Larger products use [`matmul_parallel_blocked`]. (A
/// 2x2 @ 2x2 is 8 operations; 64x64 @ 64x64 is 262,144.) Both give the same
/// answer up to rounding.
///
/// # Arguments
///
/// * `a` - Left-hand matrix, `m * k` numbers in row-major order
/// * `b` - Right-hand matrix, `k * n` numbers in row-major order
/// * `m` - Rows in `a`
/// * `n` - Columns in `b`
/// * `k` - Inner dimension
///
/// # Returns
///
/// The `m * n` result numbers in row-major order.
pub fn matmul_2d(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    if m * n * k >= 1_000 {
        matmul_parallel_blocked(a, b, m, n, k)
    } else {
        matmul_sequential(a, b, m, n, k)
    }
}

/// Multiply `count` independent pairs of matrices, one after another in memory
///
/// Each pair is `[m, k] @ [k, n]`. `a` holds `count` left matrices stored one
/// after another, `b` holds `count` right matrices, and the result holds
/// `count` product matrices. The pairs do not depend on each other, so they are
/// computed in parallel.
///
/// This is the engine for 4-D attention products: `[batch, n_heads, seq, head_dim]
/// @ [batch, n_heads, head_dim, seq]` is `count = batch * n_heads` independent
/// 2-D products, stacked. Attention computes `Q @ K^T` this way, once per head
/// per sentence: the result `[batch, heads, seq, seq]` is a table of how much
/// every token should attend to every other token.
///
/// ```text
/// count = 2 pairs, each [2, 2] @ [2, 2]
///
/// a = [ A0 (4 numbers) | A1 (4 numbers) ]
/// b = [ B0 (4 numbers) | B1 (4 numbers) ]
///            ->  result = [ A0@B0 | A1@B1 ]
/// ```
///
/// # Arguments
///
/// * `a` - `count * m * k` numbers: the left matrices, one after another
/// * `b` - `count * k * n` numbers: the right matrices, one after another
/// * `count` - how many pairs
/// * `m` - rows of each left matrix
/// * `n` - columns of each right matrix
/// * `k` - inner dimension
///
/// # Returns
///
/// The `count * m * n` result numbers: the product matrices one after another.
///
/// # Panics
///
/// Panics (index out of bounds) if `a` or `b` is shorter than the sizes say.
pub fn matmul_batched(a: &[f32], b: &[f32], count: usize, m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut result = vec![0.0; count * m * n];

    // Nothing to compute (zero pairs, rows or columns). Also avoids a chunk
    // size of 0, which `par_chunks_mut` does not allow.
    if result.is_empty() {
        return result;
    }

    // Parallelize over the pairs.
    // Each pair computes an independent m x n matrix multiplication.
    //
    // `par_chunks_mut(m * n)` cuts the output into one chunk per pair, and
    // hands each chunk to a worker thread. The chunks do not overlap, so
    // threads never write the same memory: Rust's borrow rules prove that at
    // compile time. `enumerate` supplies the chunk number `item`.
    result
        .par_chunks_mut(m * n)
        .enumerate()
        .for_each(|(item, chunk)| {
            // This pair's matrices, found by skipping the earlier pairs.
            let a_item = &a[item * m * k..(item + 1) * m * k];
            let b_item = &b[item * k * n..(item + 1) * k * n];

            // The ordinary 2-D product, written into this thread's own chunk.
            matmul_sequential_into(a_item, b_item, chunk, m, n, k);
        });

    result
}

/// Matrix multiplication
///
/// Supports:
/// - 2D × 2D: Standard matrix multiplication
/// - 4D × 4D: Batched matmul for attention (see below)
///
/// # 2D Matrix Multiplication
///
/// For `A @ B` where `A` is `[m, k]` and `B` is `[k, n]`:
/// - Result shape: `[m, n]`
/// - Each element `C[i,j] = sum(A[i,k] * B[k,j])` for all k
///
/// # Performance
///
/// - **Small matrices** (< 1K ops): Sequential computation
/// - **Large matrices** (≥ 1K ops): Parallel cache-blocked algorithm
///
/// The parallel version uses 8×8 blocks for cache efficiency and
/// parallelizes across output rows, providing 2-4x speedup on typical
/// multi-core CPUs.
///
/// # 4D Batched Matrix Multiplication
///
/// For attention computations with shape `[batch, n_heads, seq, head_dim]`
/// Processes each (batch, head) pair independently in parallel.
///
/// # Arguments
///
/// * `a` - the left tensor
/// * `b` - the right tensor
///
/// # Panics
///
/// Panics if dimensions are incompatible or unsupported
///
/// # Example
///
/// ```rust
/// use crate::Tensor;
/// use crate::tensor_functions::{matmul};
/// let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
/// let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
/// let c = matmul(&a, &b);
/// assert_eq!(c.shape, vec![2, 2]);
/// ```
///
/// # The rule, with a worked example
///
/// The INNER sizes must match (both `k`). The OUTER sizes become the
/// result shape: `[m, k] @ [k, n] -> [m, n]`.
///
/// ```text
///   A [2, 3]          B [3, 2]            C [2, 2]
///   1 2 3             1 0                 C[0][0] = 1*1 + 2*0 + 3*1 = 4
///   4 5 6      @      0 1        =        C[0][1] = 1*0 + 2*1 + 3*1 = 5
///                     1 1                 C[1][0] = 4*1 + 5*0 + 6*1 = 10
///                                         C[1][1] = 4*0 + 5*1 + 6*1 = 11
/// ```
///
/// Each cell is the dot product of a ROW of A with a COLUMN of B.
///
/// ```rust
/// use crate::Tensor;
/// use crate::tensor_functions::{matmul};
/// let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
/// let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]);
/// let c = matmul(&a, &b);
/// assert_eq!(c.data, vec![4.0, 5.0, 10.0, 11.0]);
/// ```
///
/// # 4-D shape checks
///
/// For the 4-D case the batch size, head count and inner dimension of the
/// two inputs must all match. Each mismatch panics with its own clear
/// message (rather than an "index out of bounds" from deep inside the loop).
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let (a_data, a_shape, b_data, b_shape) = (&a.data[..], &a.shape[..], &b.data[..], &b.shape[..]);

    // === 2D MATRIX MULTIPLICATION ===
    if a_shape.len() == 2 && b_shape.len() == 2 {
        // Inner sizes must match: A's columns == B's rows. The message
        // prints both shapes, because "which two shapes?" is always the
        // first question when this fails.
        assert_eq!(
            a_shape[1], b_shape[0],
            "Matrix dimensions incompatible: [{}, {}] @ [{}, {}]",
            a_shape[0], a_shape[1], b_shape[0], b_shape[1]
        );

        // Name the three sizes once, using the standard letters:
        // A is [m, k], B is [k, n], the result is [m, n].
        let m = a_shape[0];
        let n = b_shape[1];
        let k = a_shape[1];

        return Tensor::new(matmul_2d(a_data, b_data, m, n, k), vec![m, n]);
    }

    // === 4D BATCHED MATRIX MULTIPLICATION (for attention) ===
    // Shape: [batch, n_heads, seq, head_dim] @ [batch, n_heads, head_dim, seq]
    //
    // Think of it as `batch * n_heads` independent 2-D products, stacked.
    if a_shape.len() == 4 && b_shape.len() == 4 {
        let batch = a_shape[0];
        let n_heads = a_shape[1];
        let seq1 = a_shape[2]; // rows of each A matrix
        let inner_dim = a_shape[3]; // columns of A == rows of B
        let seq2 = b_shape[3]; // columns of each B matrix

        // Verify everything the index arithmetic relies on. The batch and head
        // sizes must agree, or the wrong block would be read, or the read would
        // run past the end of `b_data`.
        assert_eq!(
            b_shape[0], batch,
            "Batch sizes must match for batched matmul: {:?} @ {:?}",
            a_shape, b_shape
        );
        assert_eq!(
            b_shape[1], n_heads,
            "Head counts must match for batched matmul: {:?} @ {:?}",
            a_shape, b_shape
        );
        assert_eq!(
            b_shape[2], inner_dim,
            "Inner dimensions must match for batched matmul"
        );

        let result = matmul_batched(a_data, b_data, batch * n_heads, seq1, seq2, inner_dim);
        return Tensor::new(result, vec![batch, n_heads, seq1, seq2]);
    }

    // Anything else (3-D, 1-D, mixed ranks) is not supported.
    panic!("Unsupported matmul shapes: {:?} @ {:?}", a_shape, b_shape);
}

/// Softmax of one group of numbers: `input` in, `output` out (same length)
///
/// This is the numerically stable recipe from [`softmax`], for a
/// single row. A group that is entirely `-inf` (everything masked) gives
/// all zeros, meaning "attend to nothing", rather than `NaN`.
///
/// ```text
/// input  = [1, 2, 3]
/// max    = 3
/// exp(x - max) = [0.1353, 0.3679, 1.0]   sum = 1.5032
/// output = [0.0900, 0.2447, 0.6652]
/// ```
pub fn softmax_slice(input: &[f32], output: &mut [f32]) {
    // Largest value, for numerical stability. `fold` starting at
    // NEG_INFINITY keeps the biggest value seen so far (`f32` has no
    // `Ord`, so `iter().max()` is not available).
    let max = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Everything is -inf: `-inf - (-inf)` would be NaN. Define the result
    // as all zeros instead.
    if max == f32::NEG_INFINITY {
        output.fill(0.0);
        return;
    }

    // exp(x - max) for each number, accumulating the total as we go.
    let mut sum = 0.0;
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        let e = (x - max).exp();
        *o = e;
        sum += e;
    }

    // Normalize so the group sums to 1.
    for o in output.iter_mut() {
        *o /= sum;
    }
}


// --- 5. The tensor operations (tensors in, a new tensor out) ---

/// Softmax activation
///
/// Computes softmax along the specified axis:
///
/// ```text
/// softmax(x)[i] = exp(x[i]) / sum(exp(x[j])) for all j
/// ```
///
/// # Numerical Stability
///
/// Uses the numerically stable version:
///
/// ```text
/// softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
/// ```
///
/// Subtracting the maximum prevents overflow in exp() while producing
/// the same result (since the max factors cancel out).
///
/// # Arguments
///
/// * `t` - the tensor to normalize
/// * `axis` - Axis along which to compute softmax (use -1 for last axis)
///
/// # Performance
///
/// For the last axis (the common case), each row is computed in parallel.
/// For other axes, the tensor is split into independent outer blocks that
/// are computed in parallel.
///
/// # Example
///
/// ```rust
/// use crate::Tensor;
/// use crate::tensor_functions::{softmax};
/// let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
/// let result = softmax(&tensor, -1);
/// // Result sums to 1.0 along last dimension
/// ```
///
/// # Worked example
///
/// ```text
/// x          = [1, 2, 3]
/// max        = 3
/// x - max    = [-2, -1, 0]
/// exp(...)   = [0.1353, 0.3679, 1.0]           sum = 1.5032
/// divide     = [0.0900, 0.2447, 0.6652]        sums to 1
/// ```
///
/// Why subtract the max? `exp(300)` is about 1.9e130, bigger than the
/// largest `f32` (about 3.4e38), so a naive version overflows to
/// infinity and then `NaN`. After subtracting the max, every exponent is
/// 0 or below, so every `exp` is between 0 and 1 and the largest is
/// exactly 1.
///
/// ```rust
/// use crate::Tensor;
/// use crate::tensor_functions::{softmax};
/// let t = Tensor::new(vec![100.0, 200.0, 300.0], vec![1, 3]);
/// let p = softmax(&t, -1);
/// assert!(p.data.iter().all(|x| x.is_finite())); // no overflow
/// assert!((p.data.iter().sum::<f32>() - 1.0).abs() < 1e-6);
/// ```
///
/// # Any rank, any axis
///
/// Softmax is computed separately for each "line" of numbers along the
/// chosen axis. For 4-D attention scores `[batch, heads, seq, seq]` with
/// `axis = -1`, that is one softmax per query row, which is what attention
/// needs:
///
/// ```rust
/// use crate::Tensor;
/// use crate::tensor_functions::{softmax};
/// let scores = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
/// let p = softmax(&scores, -1);
/// // Each ROW sums to 1 (not the whole tensor).
/// assert!((p.data[0] + p.data[1] - 1.0).abs() < 1e-6);
/// assert!((p.data[2] + p.data[3] - 1.0).abs() < 1e-6);
/// ```
///
/// Softmax along axis 0 normalizes each COLUMN:
///
/// ```rust
/// use crate::Tensor;
/// use crate::tensor_functions::{softmax};
/// let t = Tensor::new(vec![1.0, 1.0, 3.0, 1.0], vec![2, 2]);
/// let p = softmax(&t, 0);
/// // Column 1 is [1, 1], so both entries are 0.5.
/// assert!((p.data[1] - 0.5).abs() < 1e-6);
/// assert!((p.data[3] - 0.5).abs() < 1e-6);
/// ```
///
/// # Fully masked rows
///
/// A group made entirely of `-inf` returns zeros instead of `NaN`. (Plain
/// softmax would compute `-inf - -inf = NaN`.) A causal mask never hides a
/// whole row, because each token can at least see itself, but padding
/// masks can.
///
/// ```rust
/// use crate::Tensor;
/// use crate::tensor_functions::{softmax};
/// let t = Tensor::new(vec![f32::NEG_INFINITY; 2], vec![1, 2]);
/// assert_eq!(softmax(&t, -1).data, vec![0.0, 0.0]);
/// ```
///
/// # Panics
///
/// Panics if `axis` is out of range for the tensor's rank.
pub fn softmax(t: &Tensor, axis: isize) -> Tensor {
    let (data, shape) = (&t.data[..], &t.shape[..]);

    // Validate and normalize the axis (-1 -> last axis).
    let axis_pos = normalize_axis(shape, axis, "softmax");

    // Nothing to compute for an empty tensor (also avoids chunks of size 0).
    if data.is_empty() {
        return Tensor::new(data.to_vec(), shape.to_vec());
    }

    // Describe the geometry. Splitting the shape around the axis:
    //   shape = [ outer dims ..., len, inner dims ... ]
    //   `len`   = how many numbers are in one softmax group
    //   `inner` = the stride between consecutive numbers of a group
    // Example: shape [2, 3, 4], axis 1 -> len 3, inner 4.
    let len = shape[axis_pos];
    let inner: usize = shape[axis_pos + 1..].iter().product();

    let mut result = vec![0.0; data.len()];

    if inner == 1 {
        // === LAST AXIS: each group is a contiguous row ===
        //
        // `par_chunks` cuts the data into rows of `len` numbers. Pairing
        // input rows with output rows lets every thread write only its own
        // rows, so no locking is needed. Works for any rank: a 4-D
        // `[b, h, s, s]` tensor is just `b*h*s` rows of `s` numbers.
        result
            .par_chunks_mut(len)
            .zip(data.par_chunks(len))
            .for_each(|(out_row, in_row)| softmax_slice(in_row, out_row));
    } else {
        // === OTHER AXES: groups are strided ===
        //
        // One outer block holds `len` rows of `inner` numbers. Group `j`
        // is column `j` of that block: elements j, j + inner, j + 2*inner...
        // Gather the column, softmax it, then scatter it back.
        let block = len * inner;
        result
            .par_chunks_mut(block)
            .zip(data.par_chunks(block))
            .for_each(|(out_block, in_block)| {
                let mut group = vec![0.0; len];
                let mut group_out = vec![0.0; len];
                for j in 0..inner {
                    for l in 0..len {
                        group[l] = in_block[l * inner + j];
                    }
                    softmax_slice(&group, &mut group_out);
                    for l in 0..len {
                        out_block[l * inner + j] = group_out[l];
                    }
                }
            });
    }

    Tensor::new(result, shape.to_vec())
}

/// Transpose two dimensions
///
/// # Arguments
///
/// * `t` - the tensor to transpose
/// * `dim1` - First dimension to swap (supports negative indexing)
/// * `dim2` - Second dimension to swap (supports negative indexing)
///
/// # Example
///
/// ```rust
/// use crate::Tensor;
/// use crate::tensor_functions::{transpose};
/// let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
/// let transposed = transpose(&tensor, 0, 1);
/// assert_eq!(transposed.shape, vec![2, 2]);
/// ```
///
/// # What transpose does
///
/// It swaps two axes. For a matrix, rows become columns. Unlike
/// [`Tensor::reshape`], the numbers are MOVED: this function builds a new
/// buffer in the new order.
///
/// ```text
/// [2, 3]                 [3, 2]
/// 1 2 3                  1 4
/// 4 5 6          ->      2 5
///                        3 6
///
/// data before: [1, 2, 3, 4, 5, 6]
/// data after:  [1, 4, 2, 5, 3, 6]
/// ```
///
/// ```rust
/// use crate::Tensor;
/// use crate::tensor_functions::{transpose};
/// let m = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
/// let t = transpose(&m, 0, 1);
/// assert_eq!(t.shape, vec![3, 2]);
/// assert_eq!(t.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
/// ```
///
/// # Any rank, any pair of axes
///
/// Three code paths, all giving the same answer as the general rule
/// "output `[.., j, .., i, ..]` = input `[.., i, .., j, ..]`":
///
/// - 2-D: a simple double loop.
/// - 4-D swapping the last two axes (either order): the attention case,
///   written out as a fast four-level loop.
/// - Everything else (3-D, other 4-D pairs, higher ranks): the general
///   stride-based path.
///
/// Swapping an axis with itself (`transpose(0, 0)`) returns an unchanged
/// copy.
///
/// ```rust
/// use crate::Tensor;
/// use crate::tensor_functions::{transpose};
/// // 3-D: swap the last two axes of a [2, 3, 4] tensor -> [2, 4, 3].
/// let t = transpose(&Tensor::arange(0, 24).reshape(&[2, 3, 4]), 1, 2);
/// assert_eq!(t.shape, vec![2, 4, 3]);
/// // Output [0, 1, 2] is input [0, 2, 1] = 0*12 + 2*4 + 1 = 9.
/// assert_eq!(t.data[0 * 12 + 1 * 3 + 2], 9.0);
/// ```
///
/// # Panics
///
/// Panics if either axis is out of range for the tensor's rank.
pub fn transpose(t: &Tensor, dim1: isize, dim2: isize) -> Tensor {
    let (data, shape) = (&t.data[..], &t.shape[..]);

    // Convert negative indices (with a range check)
    //
    // -1 means the last axis, as in Python. For ndim = 4: -1 -> 3.
    let d1 = normalize_axis(shape, dim1, "transpose");
    let d2 = normalize_axis(shape, dim2, "transpose");

    // Swapping an axis with itself changes nothing. Handling it here
    // matters: the 2-D loop below would otherwise reorder the data while
    // leaving the shape unchanged, producing an inconsistent tensor.
    if d1 == d2 {
        return Tensor::new(data.to_vec(), shape.to_vec());
    }

    // Create new shape with swapped dimensions
    //
    // [2, 3] with (0, 1) becomes [3, 2].
    let mut new_shape = shape.to_vec();
    new_shape.swap(d1, d2);

    // For 2D matrices, we can use a simple transpose
    if shape.len() == 2 {
        let rows = shape[0];
        let cols = shape[1];
        let mut result = vec![0.0; rows * cols];

        // The element in row i, column j of the input becomes row j,
        // column i of the output. The output has `rows` columns.
        //   input  [i][j] -> data[i * cols + j]
        //   output [j][i] -> data[j * rows + i]
        for i in 0..rows {
            for j in 0..cols {
                result[j * rows + i] = data[i * cols + j];
            }
        }

        return Tensor::new(result, new_shape);
    }

    // For 4D tensors transposing last two dimensions (common in attention)
    // [batch, heads, seq1, seq2] -> [batch, heads, seq2, seq1]
    //
    // Used for K^T in `Q @ K^T`.
    if shape.len() == 4 && ((d1 == 2 && d2 == 3) || (d1 == 3 && d2 == 2)) {
        let d0 = shape[0];
        let d1_dim = shape[1];
        let d2_dim = shape[2];
        let d3_dim = shape[3];

        let mut result = vec![0.0; data.len()];

        for i0 in 0..d0 {
            for i1 in 0..d1_dim {
                for i2 in 0..d2_dim {
                    for i3 in 0..d3_dim {
                        // Source layout  [d0, d1, d2, d3]: i2 outer, i3 inner.
                        // Destination layout [d0, d1, d3, d2]: i3 outer, i2 inner.
                        // Same row-major formula, with the last two sizes swapped.
                        let src_idx = ((i0 * d1_dim + i1) * d2_dim + i2) * d3_dim + i3;
                        let dst_idx = ((i0 * d1_dim + i1) * d3_dim + i3) * d2_dim + i2;
                        result[dst_idx] = data[src_idx];
                    }
                }
            }
        }

        return Tensor::new(result, new_shape);
    }

    // For higher dimensions, do full transpose with stride remapping
    //
    // The idea: for each OUTPUT flat index, find the INPUT flat index.
    //
    //   1. Split the output flat index into output coordinates, using the
    //      row-major strides of the OUTPUT shape.
    //   2. Output coordinate `k` is input coordinate `k` too, except on
    //      the two swapped axes, where the roles are exchanged. So the
    //      input flat index is the same dot product with the INPUT strides
    //      whose entries `d1` and `d2` are swapped (`swapped_strides`).
    //
    // Worked example: input shape [2, 3, 4] (strides [12, 4, 1]), swap(1, 2).
    //   output shape    = [2, 4, 3]
    //   output strides  = [12, 3, 1]
    //   swapped_strides = [12, 1, 4]
    //   output index 5: 5/12 = 0 r5;  5/3 = 1 r2;  2/1 = 2
    //                   -> output coordinates [0, 1, 2]
    //   input index = 0*12 + 1*1 + 2*4 = 9
    //   Check: output [0, 1, 2] is input [0, 2, 1] = 0*12 + 2*4 + 1*1 = 9.
    //
    // (An earlier version split the index using `swapped_strides` instead
    // of the output strides, which decoded coordinates out of range.)
    let old_strides_owned = compute_strides(shape);
    let old_strides = &old_strides_owned;
    let mut swapped_strides = old_strides.clone();
    swapped_strides.swap(d1, d2);
    let out_strides = compute_strides(&new_shape);

    let total_size = data.len();
    let mut result = vec![0.0; total_size];

    for (i, item) in result.iter_mut().enumerate().take(total_size) {
        // Compute the input flat index for output flat index `i`.
        let mut old_idx = 0;
        let mut remaining = i;

        for (dim_idx, &stride) in out_strides.iter().enumerate() {
            let coord = remaining / stride;
            remaining %= stride;
            old_idx += coord * swapped_strides[dim_idx];
        }

        *item = data[old_idx];
    }

    Tensor::new(result, new_shape)
}

/// Replace values where mask is true with given value
///
/// Used for causal masking in attention (setting future positions to -inf)
///
/// Supports broadcasting: [batch, n_heads, seq, seq] with mask [seq, seq]
///
/// # Arguments
///
/// * `t` - the tensor to mask
/// * `mask` - Boolean mask (non-zero = true)
/// * `value` - Value to fill where mask is true
///
/// # Worked example: a causal mask
///
/// A language model predicts the next token, so token `i` must not see
/// tokens after it. Set those scores to `-inf` BEFORE softmax; since
/// `exp(-inf) = 0`, those positions get exactly zero weight.
///
/// ```text
/// scores [2, 2]      mask (1 = hide)     result
/// 1 2                0 1                 1  -inf
/// 3 4                0 0                 3    4
/// ```
///
/// ```rust
/// use crate::Tensor;
/// use crate::tensor_functions::{masked_fill};
/// let scores = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
/// let mask = Tensor::new(vec![0.0, 1.0, 0.0, 0.0], vec![2, 2]);
/// let out = masked_fill(&scores, &mask, f32::NEG_INFINITY);
/// assert_eq!(out.data[0], 1.0);
/// assert_eq!(out.data[1], f32::NEG_INFINITY);
/// assert_eq!(out.data[2], 3.0);
/// assert_eq!(out.data[3], 4.0);
/// ```
///
/// The mask is an ordinary `Tensor` of floats; any non-zero value counts
/// as "true". Supported shapes: identical shapes, or a 4-D tensor with a
/// 2-D `[seq, seq]` mask applied to every batch and head.
pub fn masked_fill(t: &Tensor, mask: &Tensor, value: f32) -> Tensor {
    let (data, shape) = (&t.data[..], &t.shape[..]);
    let (mask_data, mask_shape) = (&mask.data[..], &mask.shape[..]);

    // Handle exact match
    if shape == mask_shape {
        // For each position, pick `value` if the mask is non-zero, else
        // keep the original number.
        let result = data
            .par_iter()
            .zip(mask_data)
            .map(|(&x, &m)| if m != 0.0 { value } else { x })
            .collect();
        return Tensor::new(result, shape.to_vec());
    }

    // Handle broadcasting: [batch, n_heads, seq, seq] with mask [seq, seq]
    //
    // The same [seq, seq] mask is reused for every (batch, head) block.
    if shape.len() == 4 && mask_shape.len() == 2 {
        let batch = shape[0];
        let n_heads = shape[1];
        let seq = shape[2];

        assert_eq!(mask_shape[0], seq, "Mask sequence dimension must match");
        assert_eq!(mask_shape[1], seq, "Mask sequence dimension must match");

        let mut result = Vec::with_capacity(data.len());
        // The loop variables `_b` and `_h` are unused (leading underscore):
        // we only need to repeat the inner work batch * n_heads times.
        for _b in 0..batch {
            for _h in 0..n_heads {
                for i in 0..seq {
                    for j in 0..seq {
                        // The mask position depends only on (i, j).
                        let mask_idx = i * seq + j;
                        // `result.len()` is the number of values pushed so
                        // far, which equals the flat index of the cell we
                        // are about to write. Walking in row-major order,
                        // that doubles as the input index.
                        let self_idx = result.len();
                        result.push(if mask_data[mask_idx] != 0.0 {
                            value
                        } else {
                            data[self_idx]
                        });
                    }
                }
            }
        }
        return Tensor::new(result, shape.to_vec());
    }

    panic!(
        "Unsupported broadcast for masked_fill: {:?} with mask {:?}",
        shape, mask_shape
    );
}

/// Compute mean along an axis
///
/// # Arguments
///
/// * `t` - the tensor to average
/// * `axis` - Axis along which to compute mean (use -1 for last axis)
/// * `keepdim` - Whether to keep the reduced dimension (size 1)
///
/// # What it computes
///
/// The average of the numbers along one axis, which removes that axis.
/// Works for any rank and any axis.
///
/// ```text
/// x [2, 3]      mean(-1, false) -> shape [2]       [2, 5]
/// 1 2 3         mean(-1, true)  -> shape [2, 1]    [[2], [5]]
/// 4 5 6         mean(0, false)  -> shape [3]       [2.5, 3.5, 4.5]
/// ```
///
/// `keepdim = true` leaves a size-1 axis in place of the removed one.
/// That is what lets the result broadcast back against the original in
/// [`sub`] and [`div`].
///
/// ```rust
/// use crate::Tensor;
/// use crate::tensor_functions::{mean};
/// let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
/// let m = mean(&x, -1, true);
/// assert_eq!(m.shape, vec![2, 1]);
/// assert_eq!(m.data, vec![2.0, 5.0]);
/// assert_eq!(mean(&x, 0, false).data, vec![2.5, 3.5, 4.5]);
/// ```
///
/// # How the axis is handled
///
/// The shape is split around the axis into `outer` (dimensions before),
/// `len` (the axis itself) and `inner` (dimensions after). Each output
/// number averages `len` input numbers that are `inner` apart in memory.
///
/// ```text
/// shape [2, 3, 4], axis 1:   outer = 2, len = 3, inner = 4
/// output element (o, i) averages data[(o*3 + l)*4 + i] for l = 0..3
/// ```
///
/// The mean of an axis of length 0 is `NaN` (0 / 0).
///
/// # Panics
///
/// Panics if `axis` is out of range for the tensor's rank.
pub fn mean(t: &Tensor, axis: isize, keepdim: bool) -> Tensor {
    let (data, shape) = (&t.data[..], &t.shape[..]);

    // Convert a negative axis to a normal index (with a range check).
    let axis_pos = normalize_axis(shape, axis, "mean");

    // Split the shape around the axis (see the diagram above).
    let len = shape[axis_pos];
    let outer: usize = shape[..axis_pos].iter().product();
    let inner: usize = shape[axis_pos + 1..].iter().product();

    // One output number per (outer, inner) pair, computed in parallel.
    // `j` numbers the outputs 0..outer*inner; decode it into (o, i).
    // If `inner` is 0 the range is empty, so `j / inner` never runs.
    let result: Vec<f32> = (0..outer * inner)
        .into_par_iter()
        .map(|j| {
            let o = j / inner;
            let i = j % inner;
            // Index of the first number to average; each next one is
            // `inner` further on.
            let base = o * len * inner + i;
            let sum: f32 = (0..len).map(|l| data[base + l * inner]).sum();
            // `as f32` converts the count to a float before dividing.
            sum / len as f32
        })
        .collect();

    // `keepdim` decides whether the averaged axis stays as size 1.
    let mut new_shape = shape.to_vec();
    if keepdim {
        new_shape[axis_pos] = 1;
    } else {
        new_shape.remove(axis_pos);
    }
    Tensor::new(result, new_shape)
}

/// Compute variance along an axis
///
/// # Arguments
///
/// * `t` - the tensor to measure
/// * `axis` - Axis along which to compute variance (typically -1 for last dimension)
/// * `keepdim` - Whether to keep the reduced dimension
///
/// # Example
///
/// ```rust
/// use crate::Tensor;
/// use crate::tensor_functions::{var};
/// let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
/// let var = var(&x, -1, true); // Variance along last dimension
/// assert_eq!(var.shape, vec![2, 1]);
/// ```
///
/// # What it computes
///
/// Variance measures how spread out the numbers are around their mean:
///
/// ```text
/// variance = sum( (x - mean)^2 ) / n
///
/// x = [1, 2, 3]   mean = 2
/// gaps:  -1, 0, 1      squared: 1, 0, 1      sum = 2
/// variance = 2 / 3 = 0.6667
/// ```
///
/// This divides by `n` (the POPULATION variance), not `n - 1` (the
/// sample variance). Layer normalization conventionally uses `n`.
/// The standard deviation is `var(..).sqrt()`.
///
/// ```rust
/// use crate::Tensor;
/// use crate::tensor_functions::{var};
/// let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
/// let v = var(&x, -1, false);
/// assert!((v.data[0] - 2.0 / 3.0).abs() < 1e-6);
/// ```
///
/// # Any rank, any axis
///
/// Like [`mean`], this works for any rank and any axis, and the
/// output positions are computed in parallel. The shape is split around
/// the axis into `outer` (dimensions before), `len` (the axis itself) and
/// `inner` (dimensions after); each output number uses `len` input numbers
/// that are `inner` apart in memory.
///
/// ```rust
/// use crate::Tensor;
/// use crate::tensor_functions::{var};
/// // Down the columns (axis 0) of [[1, 2], [3, 6]]:
/// //   column 0 = [1, 3] -> mean 2, variance 1
/// //   column 1 = [2, 6] -> mean 4, variance 4
/// let x = Tensor::new(vec![1.0, 2.0, 3.0, 6.0], vec![2, 2]);
/// assert_eq!(var(&x, 0, false).data, vec![1.0, 4.0]);
/// ```
///
/// The variance of an empty axis is `NaN` (0 / 0).
///
/// # Panics
///
/// Panics if `axis` is out of range for the tensor's rank.
pub fn var(t: &Tensor, axis: isize, keepdim: bool) -> Tensor {
    let (data, shape) = (&t.data[..], &t.shape[..]);

    // Convert a negative axis to a normal index (with a range check).
    let axis_pos = normalize_axis(shape, axis, "var");

    // Split the shape around the axis (see the diagram in the docs).
    let len = shape[axis_pos];
    let outer: usize = shape[..axis_pos].iter().product();
    let inner: usize = shape[axis_pos + 1..].iter().product();

    // One output number per (outer, inner) pair, computed in parallel.
    // `j` numbers the outputs 0..outer*inner; decode it into (o, i).
    // If `inner` is 0 the range is empty, so `j / inner` never runs.
    let result: Vec<f32> = (0..outer * inner)
        .into_par_iter()
        .map(|j| {
            let o = j / inner;
            let i = j % inner;
            // Index of the first number of this group; each next one is
            // `inner` further on.
            let base = o * len * inner + i;

            // Compute mean for this group
            let mean: f32 = (0..len).map(|l| data[base + l * inner]).sum::<f32>()
                / len as f32;

            // Compute variance
            // Average of squared distances from the mean. `powi(2)`
            // raises to an integer power (2), cheaper than `powf`.
            (0..len)
                .map(|l| (data[base + l * inner] - mean).powi(2))
                .sum::<f32>()
                / len as f32
        })
        .collect();

    // Shape of the result: keep the axis as size 1, or drop it.
    //   keepdim: [2, 3] -> [2, 1]      not keepdim: [2, 3] -> [2]
    let mut new_shape = shape.to_vec();
    if keepdim {
        new_shape[axis_pos] = 1;
    } else {
        new_shape.remove(axis_pos);
    }
    Tensor::new(result, new_shape)
}

/// Concatenate two tensors along the first dimension (sequence dimension)
///
/// Used for KV caching: concatenates cached K/V with new K/V.
///
/// # Arguments
///
/// * `a` - the first tensor
/// * `b` - the second tensor, joined after `a`
/// * `b` - Tensor to concatenate with (must have matching shape except first dim)
///
/// # Example
///
/// ```rust
/// use crate::Tensor;
/// use crate::tensor_functions::{concat};
/// let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
/// let b = Tensor::new(vec![5.0, 6.0], vec![1, 2]);
/// let c = concat(&a, &b);
/// assert_eq!(c.shape, vec![3, 2]);
/// assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// ```
///
/// # Why this is just "append the data"
///
/// In row-major order, the first dimension is the OUTERMOST: each step
/// along it is one whole contiguous block. So stacking tensors along
/// dimension 0 means placing one block after the other in memory.
///
/// ```text
/// a [2, 2]: 1 2 | 3 4      b [1, 2]: 5 6
/// a ++ b   -> [3, 2]:  1 2 | 3 4 | 5 6      (data: [1,2,3,4,5,6])
/// ```
///
/// Joining along an INNER dimension would not be this simple, because
/// the numbers would have to interleave. That is why this function is
/// limited to the first dimension.
///
/// # KV caching
///
/// When a model generates text one token at a time, it keeps the keys and
/// values computed for earlier tokens. Each step it computes one new
/// key/value row and appends it with `concat`, rather than recomputing
/// the whole history.
///
/// # Panics
///
/// Panics if the ranks differ or any dimension other than the first
/// differs.
pub fn concat(a: &Tensor, b: &Tensor) -> Tensor {
    let (a_data, a_shape, b_data, b_shape) = (&a.data[..], &a.shape[..], &b.data[..], &b.shape[..]);

    // Same rank?
    assert_eq!(
        a_shape.len(),
        b_shape.len(),
        "Tensors must have same number of dimensions for concat"
    );

    // Check that all dimensions except the first match
    //
    // The loop starts at 1 to skip dimension 0, the one allowed to differ.
    for i in 1..a_shape.len() {
        assert_eq!(
            a_shape[i], b_shape[i],
            "All dimensions except first must match for concat"
        );
    }

    // Concatenate along first dimension
    //
    // Only dimension 0 grows: [2, 2] and [1, 2] -> [3, 2].
    let mut new_shape = a_shape.to_vec();
    new_shape[0] += b_shape[0];

    // Reserve the exact space up front so the Vec does not reallocate.
    let mut data = Vec::with_capacity(a_data.len() + b_data.len());
    data.extend_from_slice(&a_data);
    data.extend_from_slice(&b_data);

    Tensor::new(data, new_shape)
}

/// Element-wise addition with broadcasting support
///
/// Supports exact matches and general broadcasting (NumPy rules), which
/// covers the patterns common in transformers:
///
/// 1. **Exact match**: Same shape
/// 2. **Broadcast last dim**: `[*, n] + [n]` (e.g., adding bias)
/// 3. **Broadcast batch**: `[batch, seq, dim] + [seq, dim]`
/// 4. **Keepdim**: `[.., n] + [.., 1]`
///
/// The smaller tensor may be on either side: `add(&bias, &matrix)` and
/// `add(&matrix, &bias)` give the same result.
///
/// # Arguments
///
/// * `a` - the left tensor
/// * `b` - Tensor to add (may have different shape if broadcasting applies)
///
/// # Example
///
/// ```rust
/// use crate::Tensor;
/// use crate::tensor_functions::{add};
/// let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
/// let b = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
/// let c = add(&a, &b);
/// assert_eq!(c.data, vec![2.0, 3.0, 4.0, 5.0]);
/// ```
///
/// # Broadcasting, explained
///
/// Broadcasting lets a smaller tensor be combined with a larger one by
/// REPEATING it, without storing the repeats. Shapes are lined up from the
/// right; each pair of sizes must be equal, or one of them must be 1 (a
/// missing leading dimension counts as 1). A repeated dimension is read
/// with stride 0 (see `broadcast_strides`).
///
/// ```text
/// bias: [0.1 0.2 0.3]  (shape [3])    matrix: [2, 3], flat index i = 0..5
///
/// i:             0    1    2    3    4    5
/// i % 3:         0    1    2    0    1    2     <- which bias value
/// matrix[i]:     1    2    3    4    5    6
/// result:       1.1  2.2  3.3  4.1  5.2  6.3
/// ```
///
/// ```rust
/// use crate::Tensor;
/// use crate::tensor_functions::{add};
/// let m = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
/// let bias = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
/// let r = add(&m, &bias);
/// assert_eq!(r.data, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
///
/// // The smaller tensor may also be on the left.
/// assert_eq!(add(&bias, &m).data, r.data);
/// ```
///
/// # Panics
///
/// Panics if the shapes cannot be broadcast together.
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let (data, shape) = broadcast_zip(&a.data, &a.shape, &b.data, &b.shape, "add", |x, y| x + y);
    Tensor::new(data, shape)
}

/// Element-wise subtraction with broadcasting
///
/// See `add()` for the broadcasting rules. `self` is always the left
/// operand: `sub(&a, &b)` is `a - b` whichever tensor is smaller.
///
/// The `keepdim` pattern works, which is the shape that
/// `mean(.., keepdim = true)` returns, so `sub(&x, &mean(&x, -1, true))`
/// ("subtract each row's mean") works. A bias vector `[n]` works too.
///
/// ```text
/// x [2, 3]            mean [2, 1]         x - mean
/// 1 2 3               2                   -1  0  1
/// 4 5 6               5                   -1  0  1
/// ```
///
/// ```rust
/// use crate::Tensor;
/// use crate::tensor_functions::{mean, sub};
/// let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
/// let centered = sub(&x, &mean(&x, -1, true));
/// assert_eq!(centered.data, vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]);
/// ```
///
/// # Arguments
///
/// * `a` - the left tensor (the one subtracted FROM)
/// * `b` - the right tensor
///
/// # Panics
///
/// Panics if the shapes cannot be broadcast together.
pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    let (data, shape) = broadcast_zip(&a.data, &a.shape, &b.data, &b.shape, "sub", |x, y| x - y);
    Tensor::new(data, shape)
}

/// Element-wise multiplication with broadcasting
///
/// See `add()` for the broadcasting rules.
///
/// This is the element-wise (Hadamard) product: cell `[i][j]` times cell
/// `[i][j]`. It is NOT matrix multiplication; for that, use
/// [`matmul`].
///
/// ```rust
/// use crate::Tensor;
/// use crate::tensor_functions::{mul};
/// let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
/// let b = Tensor::new(vec![10.0, 100.0], vec![2]);
/// assert_eq!(mul(&a, &b).data, vec![10.0, 200.0, 30.0, 400.0]);
/// ```
///
/// # Arguments
///
/// * `a` - the left tensor
/// * `b` - the right tensor
///
/// # Panics
///
/// Panics if the shapes cannot be broadcast together.
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let (data, shape) = broadcast_zip(&a.data, &a.shape, &b.data, &b.shape, "mul", |x, y| x * y);
    Tensor::new(data, shape)
}

/// Element-wise division with broadcasting
///
/// See `add()` for the broadcasting rules, and `sub()` for the `keepdim`
/// pattern. Dividing a row by its own standard deviation, as in layer
/// normalization, uses that form.
///
/// Dividing by zero follows `f32` rules: `x / 0.0` is `inf` (or `NaN`
/// when `x` is also 0), and no panic occurs.
///
/// # Arguments
///
/// * `a` - the left tensor (the numerator)
/// * `b` - the right tensor
///
/// # Panics
///
/// Panics if the shapes cannot be broadcast together.
pub fn div(a: &Tensor, b: &Tensor) -> Tensor {
    let (data, shape) = broadcast_zip(&a.data, &a.shape, &b.data, &b.shape, "div", |x, y| x / y);
    Tensor::new(data, shape)
}

// Tests for the pure functions. They call the functions directly, with plain
// slices and shapes. Tests of the Tensor type itself are in `tensor.rs`.
#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Pure functions: plain slices and shapes in, plain data out.
    // No Tensor is needed.
    // ------------------------------------------------------------------

    /// Assert two float slices are equal within a small tolerance.
    fn assert_close(actual: &[f32], expect: &[f32]) {
        assert_eq!(actual.len(), expect.len(), "length differs");
        for (i, (a, e)) in actual.iter().zip(expect).enumerate() {
            assert!((a - e).abs() < 1e-5, "index {}: {} vs {}", i, a, e);
        }
    }

    #[test]
    fn test_compute_strides() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(compute_strides(&[5]), vec![1]);
        let empty: Vec<usize> = vec![];
        assert_eq!(compute_strides(&[]), empty);
    }

    #[test]
    fn test_normalize_axis() {
        let shape = [2, 3, 4];
        assert_eq!(normalize_axis(&shape, 0, "test"), 0);
        assert_eq!(normalize_axis(&shape, 2, "test"), 2);
        assert_eq!(normalize_axis(&shape, -1, "test"), 2);
        assert_eq!(normalize_axis(&shape, -3, "test"), 0);
    }

    #[test]
    #[should_panic(expected = "test: axis 3 is out of range for shape [2, 3, 4]")]
    fn test_normalize_axis_too_large() {
        let _ = normalize_axis(&[2, 3, 4], 3, "test");
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_normalize_axis_too_negative() {
        let _ = normalize_axis(&[2, 3, 4], -4, "test");
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_normalize_axis_scalar_has_no_axes() {
        let _ = normalize_axis(&[], 0, "test");
    }

    #[test]
    fn test_broadcast_shapes() {
        assert_eq!(broadcast_shapes(&[2, 3], &[3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shapes(&[3], &[2, 3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shapes(&[2, 1, 3], &[1, 2, 1]), Some(vec![2, 2, 3]));
        assert_eq!(broadcast_shapes(&[2, 3], &[4, 3]), None);
        assert_eq!(broadcast_shapes(&[], &[2]), Some(vec![2]));
    }

    #[test]
    fn test_broadcast_strides_vector_into_matrix() {
        // A bias of shape [3] read as shape [2, 3]: the row step stays put.
        let actual = broadcast_strides(&[3], &[1], &[2, 3]);
        let expect = vec![0, 1];
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_broadcast_strides_keepdim_column() {
        // A [2, 1] column read as [2, 3]: the column step stays put.
        let actual = broadcast_strides(&[2, 1], &[1, 1], &[2, 3]);
        let expect = vec![1, 0];
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_broadcast_strides_same_shape_keeps_strides() {
        let actual = broadcast_strides(&[2, 3], &[3, 1], &[2, 3]);
        let expect = vec![3, 1];
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_broadcast_zip_same_shape() {
        let (actual_data, actual_shape) =
            broadcast_zip(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &[10.0, 20.0, 30.0, 40.0], &[2, 2], "add", |a, b| a + b);
        assert_eq!(actual_data, vec![11.0, 22.0, 33.0, 44.0]);
        assert_eq!(actual_shape, vec![2, 2]);
    }

    #[test]
    fn test_broadcast_zip_bias_either_order() {
        let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias = [10.0, 20.0, 30.0];
        let expect_data = vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0];

        let (data, shape) = broadcast_zip(&matrix, &[2, 3], &bias, &[3], "add", |a, b| a + b);
        assert_eq!(data, expect_data);
        assert_eq!(shape, vec![2, 3]);

        // The smaller array on the left gives the same answer.
        let (data, shape) = broadcast_zip(&bias, &[3], &matrix, &[2, 3], "add", |a, b| a + b);
        assert_eq!(data, expect_data);
        assert_eq!(shape, vec![2, 3]);
    }

    #[test]
    fn test_broadcast_zip_keeps_operand_order() {
        // f always gets the FIRST array's number first, even when it is smaller.
        let bias = [1.0, 2.0, 3.0];
        let matrix = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let (actual, _) = broadcast_zip(&bias, &[3], &matrix, &[2, 3], "sub", |a, b| a - b);
        let expect = vec![-9.0, -18.0, -27.0, -39.0, -48.0, -57.0];
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_broadcast_zip_both_sides_expand() {
        // [2, 1] + [1, 3] -> [2, 3]
        let (actual_data, actual_shape) =
            broadcast_zip(&[10.0, 20.0], &[2, 1], &[1.0, 2.0, 3.0], &[1, 3], "add", |a, b| a + b);
        assert_eq!(actual_data, vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0]);
        assert_eq!(actual_shape, vec![2, 3]);
    }

    #[test]
    #[should_panic(expected = "Cannot broadcast shapes [2, 3] and [4, 3] for add")]
    fn test_broadcast_zip_incompatible_shapes_panics() {
        let a = [0.0; 6];
        let b = [0.0; 12];
        let _ = broadcast_zip(&a, &[2, 3], &b, &[4, 3], "add", |x, y| x + y);
    }

    #[test]
    fn test_matmul_inner_simd_matches_doc_example() {
        let mut actual = [5.0, 5.0, 5.0];
        matmul_inner_simd(2.0, &[1.0, 10.0, 100.0], &mut actual);
        let expect = [7.0, 25.0, 205.0];
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_matmul_parallel_blocked_matches_naive_loops() {
        // Sizes that are NOT multiples of the 8x8 block, so partial blocks run.
        let (m, k, n) = (21, 13, 17);
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.5).collect();
        let mut expect = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    expect[i * n + j] += a[i * k + l] * b[l * n + j];
                }
            }
        }
        let actual = matmul_parallel_blocked(&a, &b, m, n, k);
        assert_close(&actual, &expect);
    }

    #[test]
    fn test_matmul_parallel_blocked_identity() {
        let a: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let actual = matmul_parallel_blocked(&a, &identity, 3, 3, 3);
        assert_eq!(actual, a);
    }

    #[test]
    fn test_softmax_slice_known_values() {
        let input = [1.0, 2.0, 3.0];
        let mut actual = [0.0; 3];
        softmax_slice(&input, &mut actual);
        let expect = [0.0900306, 0.2447285, 0.6652410];
        assert_close(&actual, &expect);
    }

    #[test]
    fn test_softmax_slice_is_stable_for_large_values() {
        let input = [100.0, 200.0, 300.0];
        let mut actual = [0.0; 3];
        softmax_slice(&input, &mut actual);
        assert!(actual.iter().all(|x| x.is_finite()));
        assert_close(&actual, &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_softmax_slice_fully_masked_is_zeros() {
        let input = [f32::NEG_INFINITY, f32::NEG_INFINITY];
        let mut actual = [9.0; 2];
        softmax_slice(&input, &mut actual);
        let expect = [0.0, 0.0];
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_softmax_slice_partly_masked() {
        let input = [1.0, f32::NEG_INFINITY];
        let mut actual = [0.0; 2];
        softmax_slice(&input, &mut actual);
        let expect = [1.0, 0.0];
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_matmul_sequential_known_values() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let b = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // [3, 2]
        let actual = matmul_sequential(&a, &b, 2, 2, 3);
        let expect = vec![4.0, 5.0, 10.0, 11.0];
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_matmul_sequential_into_overwrites_the_buffer() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let identity = [1.0, 0.0, 0.0, 1.0];
        let mut actual = [9.0; 4]; // stale contents must not leak into the answer
        matmul_sequential_into(&a, &identity, &mut actual, 2, 2, 2);
        let expect = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_matmul_2d_small_and_large_agree_with_sequential() {
        // 2x2x2 = 8 operations: sequential path. 20x20x20 = 8000: blocked path.
        for size in [2usize, 20] {
            let a: Vec<f32> = (0..size * size).map(|i| (i % 5) as f32 - 2.0).collect();
            let b: Vec<f32> = (0..size * size).map(|i| (i % 3) as f32 + 0.5).collect();
            let actual = matmul_2d(&a, &b, size, size, size);
            let expect = matmul_sequential(&a, &b, size, size, size);
            assert_close(&actual, &expect);
        }
    }

    #[test]
    fn test_matmul_batched_each_pair_is_its_own_product() {
        // Two pairs of [2, 2] matrices. Pair 0 multiplies by the identity,
        // pair 1 by twice the identity.
        let a = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
        let actual = matmul_batched(&a, &b, 2, 2, 2, 2);
        let expect = vec![1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0];
        assert_eq!(actual, expect);
    }

    #[test]
    fn test_matmul_batched_non_square() {
        // 3 pairs of [2, 4] @ [4, 5], all ones: every result number is 4.
        let a = vec![1.0; 3 * 2 * 4];
        let b = vec![1.0; 3 * 4 * 5];
        let actual = matmul_batched(&a, &b, 3, 2, 5, 4);
        assert_eq!(actual.len(), 3 * 2 * 5);
        assert!(actual.iter().all(|&x| x == 4.0));
    }

    #[test]
    fn test_matmul_batched_matches_sequential_per_pair() {
        let (count, m, k, n) = (4, 3, 5, 2);
        let a: Vec<f32> = (0..count * m * k).map(|i| (i % 7) as f32 - 3.0).collect();
        let b: Vec<f32> = (0..count * k * n).map(|i| (i % 4) as f32).collect();
        let actual = matmul_batched(&a, &b, count, m, n, k);
        for item in 0..count {
            let expect = matmul_sequential(
                &a[item * m * k..(item + 1) * m * k],
                &b[item * k * n..(item + 1) * k * n],
                m,
                n,
                k,
            );
            assert_eq!(&actual[item * m * n..(item + 1) * m * n], &expect[..]);
        }
    }

    #[test]
    fn test_matmul_batched_empty() {
        let actual = matmul_batched(&[], &[], 0, 2, 2, 2);
        let expect: Vec<f32> = vec![];
        assert_eq!(actual, expect);
    }

    // ------------------------------------------------------------------
    // Tensor operations (moved here from tensor.rs)
    // ------------------------------------------------------------------

    /// Reference transpose: build the output element by element from the
    /// definition "output[.., j, .., i, ..] = input[.., i, .., j, ..]".
    fn reference_transpose(t: &Tensor, d1: usize, d2: usize) -> Tensor {
        let mut new_shape = t.shape.clone();
        new_shape.swap(d1, d2);
        let out_strides = compute_strides(&new_shape);
        let mut data = vec![0.0; t.data.len()];
        for (i, slot) in data.iter_mut().enumerate() {
            // Output coordinates.
            let mut rem = i;
            let mut coords = vec![0; new_shape.len()];
            for k in 0..new_shape.len() {
                coords[k] = rem / out_strides[k];
                rem %= out_strides[k];
            }
            // Swap the two coordinates to get the input coordinates.
            coords.swap(d1, d2);
            let idx: usize = coords.iter().zip(&t.strides).map(|(c, s)| c * s).sum();
            *slot = t.data[idx];
        }
        Tensor::new(data, new_shape)
    }

    // ---------- softmax ----------

    #[test]
    fn test_softmax_2d_rows() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 1.0, 1.0, 1.0], vec![2, 3]);
        let p = softmax(&t, -1);
        assert_close(&p.data[..3], &[0.0900306, 0.2447285, 0.6652410]);
        assert_close(&p.data[3..], &[1.0 / 3.0; 3]);
    }

    #[test]
    fn test_softmax_4d_is_per_row_not_global() {
        // The old code softmaxed all 32 numbers together for 4-D input.
        let t = Tensor::arange(0, 32).reshape(&[2, 2, 2, 4]);
        let p = softmax(&t, -1);
        assert_eq!(p.shape, vec![2, 2, 2, 4]);
        for row in p.data.chunks(4) {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row sums to {}", sum);
        }
        // Must equal the 2-D result on the flattened rows.
        let flat = softmax(&t.reshape(&[8, 4]), -1);
        assert_close(&p.data, &flat.data);
    }

    #[test]
    fn test_softmax_3d_last_axis() {
        let t = Tensor::arange(0, 12).reshape(&[2, 3, 2]);
        let p = softmax(&t, 2);
        for row in p.data.chunks(2) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_axis_0_normalizes_columns() {
        let t = Tensor::new(vec![1.0, 1.0, 3.0, 1.0], vec![2, 2]);
        let p = softmax(&t, 0);
        // Column 0 is [1, 3]; column 1 is [1, 1].
        assert_close(&p.data, &[0.1192029, 0.5, 0.8807971, 0.5]);
    }

    #[test]
    fn test_softmax_middle_axis_matches_transposed_last_axis() {
        // Cross-check the strided path against the contiguous path:
        // softmax over axis 1 == transpose, softmax over last axis, transpose back.
        let t = Tensor::new(
            (0..24).map(|i| ((i * 7) % 11) as f32 * 0.3).collect(),
            vec![2, 3, 4],
        );
        let direct = softmax(&t, 1);
        let via_transpose = transpose(&softmax(&transpose(&t, 1, 2), -1), 1, 2);
        assert_eq!(direct.shape, via_transpose.shape);
        assert_close(&direct.data, &via_transpose.data);
    }

    #[test]
    fn test_softmax_negative_axis_equals_positive() {
        let t = Tensor::arange(0, 24).reshape(&[2, 3, 4]);
        assert_close(&softmax(&t, -2).data, &softmax(&t, 1).data);
    }

    #[test]
    fn test_softmax_large_values_are_stable() {
        let t = Tensor::new(vec![100.0, 200.0, 300.0], vec![1, 3]);
        let p = softmax(&t, -1);
        assert!(p.data.iter().all(|x| x.is_finite()));
        assert_close(&p.data, &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_softmax_fully_masked_row_is_zeros() {
        let t = Tensor::new(
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY],
            vec![2, 2],
        );
        let p = softmax(&t, -1);
        assert_eq!(p.data, vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_softmax_after_causal_mask() {
        // The real use: mask the future, then softmax. Row 0 sees only itself.
        let scores = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let mask = Tensor::new(vec![0.0, 1.0, 0.0, 0.0], vec![2, 2]);
        let p = softmax(&masked_fill(&scores, &mask, f32::NEG_INFINITY), -1);
        assert_eq!(p.data[0], 1.0);
        assert_eq!(p.data[1], 0.0);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_softmax_axis_out_of_range() {
        let _ = softmax(&Tensor::zeros(vec![2, 2]), 2);
    }

    // ---------- transpose ----------

    #[test]
    fn test_transpose_2d() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let r = transpose(&t, 0, 1);
        assert_eq!(r.shape, vec![3, 2]);
        assert_eq!(r.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_same_axis_is_identity() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let r = transpose(&t, 0, 0);
        assert_eq!(r.shape, vec![2, 3]);
        assert_eq!(r.data, t.data);
        let r = transpose(&t, -1, 1);
        assert_eq!(r.shape, vec![2, 3]);
        assert_eq!(r.data, t.data);
    }

    #[test]
    fn test_transpose_3d_all_pairs_match_reference() {
        // The old code panicked on every one of these.
        let t = Tensor::arange(0, 24).reshape(&[2, 3, 4]);
        for (d1, d2) in [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)] {
            let got = transpose(&t, d1 as isize, d2 as isize);
            let want = reference_transpose(&t, d1, d2);
            assert_eq!(got.shape, want.shape, "shape for ({}, {})", d1, d2);
            assert_eq!(got.data, want.data, "data for ({}, {})", d1, d2);
        }
    }

    #[test]
    fn test_transpose_3d_worked_example() {
        // The example in the code comment: shape [2, 3, 4], swap(1, 2).
        let t = Tensor::arange(0, 24).reshape(&[2, 3, 4]);
        let r = transpose(&t, 1, 2);
        assert_eq!(r.shape, vec![2, 4, 3]);
        // Output [0, 1, 2] is input [0, 2, 1] = 9, at output flat index 5.
        assert_eq!(r.data[5], 9.0);
    }

    #[test]
    fn test_transpose_4d_all_pairs_match_reference() {
        let t = Tensor::arange(0, 120).reshape(&[2, 3, 4, 5]);
        for d1 in 0..4 {
            for d2 in 0..4 {
                let got = transpose(&t, d1 as isize, d2 as isize);
                let want = if d1 == d2 { t.clone() } else { reference_transpose(&t, d1, d2) };
                assert_eq!(got.shape, want.shape, "shape for ({}, {})", d1, d2);
                assert_eq!(got.data, want.data, "data for ({}, {})", d1, d2);
            }
        }
    }

    #[test]
    fn test_transpose_twice_is_identity() {
        let t = Tensor::arange(0, 60).reshape(&[3, 4, 5]);
        let back = transpose(&transpose(&t, 0, 2), 0, 2);
        assert_eq!(back.shape, t.shape);
        assert_eq!(back.data, t.data);
    }

    #[test]
    fn test_transpose_negative_axes() {
        let t = Tensor::arange(0, 24).reshape(&[2, 3, 4]);
        assert_eq!(transpose(&t, -1, -2).data, transpose(&t, 2, 1).data);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_transpose_axis_out_of_range() {
        let _ = transpose(&Tensor::zeros(vec![2, 3]), 0, 2);
    }

    // ---------- broadcasting ----------

    #[test]
    fn test_add_bias_either_order() {
        let m = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
        let want = vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
        assert_eq!(add(&m, &b).data, want);
        // The old code panicked here.
        let r = add(&b, &m);
        assert_eq!(r.shape, vec![2, 3]);
        assert_eq!(r.data, want);
    }

    #[test]
    fn test_sub_bias_keeps_operand_order() {
        let m = Tensor::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        // matrix - bias
        assert_eq!(sub(&m, &b).data, vec![9.0, 18.0, 27.0, 39.0, 48.0, 57.0]);
        // bias - matrix (smaller tensor on the LEFT must still be the left operand)
        assert_eq!(sub(&b, &m).data, vec![-9.0, -18.0, -27.0, -39.0, -48.0, -57.0]);
    }

    #[test]
    fn test_div_bias_vector() {
        let m = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2]);
        let b = Tensor::new(vec![2.0, 4.0], vec![2]);
        assert_eq!(div(&m, &b).data, vec![1.0, 1.0, 3.0, 2.0]);
    }

    #[test]
    fn test_mul_batch_by_2d() {
        // [batch, seq, dim] * [seq, dim]: the old code had no such case.
        let x = Tensor::new(vec![1.0; 12], vec![2, 2, 3]);
        let w = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let r = mul(&x, &w);
        assert_eq!(r.shape, vec![2, 2, 3]);
        assert_eq!(&r.data[..6], &w.data[..]);
        assert_eq!(&r.data[6..], &w.data[..]);
    }

    #[test]
    fn test_add_batch_by_2d() {
        let x = Tensor::new(vec![0.0; 12], vec![2, 2, 3]);
        let p = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let r = add(&x, &p);
        assert_eq!(&r.data[..6], &p.data[..]);
        assert_eq!(&r.data[6..], &p.data[..]);
    }

    #[test]
    fn test_sub_keepdim_mean() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let centered = sub(&x, &mean(&x, -1, true));
        assert_eq!(centered.data, vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_broadcast_both_sides_expand() {
        // [2, 1] + [1, 3] -> [2, 3] (an "outer sum").
        let a = Tensor::new(vec![10.0, 20.0], vec![2, 1]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let r = add(&a, &b);
        assert_eq!(r.shape, vec![2, 3]);
        assert_eq!(r.data, vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0]);
    }

    #[test]
    fn test_broadcast_with_scalar_tensor() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let s = Tensor::new(vec![10.0], vec![]);
        assert_eq!(add(&a, &s).data, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    #[should_panic(expected = "Cannot broadcast shapes")]
    fn test_add_incompatible_shapes_panics() {
        let a = Tensor::zeros(vec![2, 3]);
        let b = Tensor::zeros(vec![4, 3]);
        let _ = add(&a, &b);
    }

    // ---------- mean ----------

    #[test]
    fn test_mean_2d_axes() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(mean(&x, -1, false).data, vec![2.0, 5.0]);
        assert_eq!(mean(&x, 0, false).data, vec![2.5, 3.5, 4.5]);
        assert_eq!(mean(&x, -1, true).shape, vec![2, 1]);
        assert_eq!(mean(&x, 0, true).shape, vec![1, 3]);
    }

    #[test]
    fn test_mean_4d() {
        // The old code panicked on 4-D input.
        let x = Tensor::arange(0, 32).reshape(&[2, 2, 2, 4]);
        let last = mean(&x, -1, false);
        assert_eq!(last.shape, vec![2, 2, 2]);
        // First row is [0, 1, 2, 3] -> mean 1.5.
        assert_eq!(last.data[0], 1.5);
        let keep = mean(&x, -1, true);
        assert_eq!(keep.shape, vec![2, 2, 2, 1]);
        // Middle axis: output (0,0,0,0) averages x[0,0,0,0] and x[0,1,0,0] = (0 + 8) / 2.
        let mid = mean(&x, 1, false);
        assert_eq!(mid.shape, vec![2, 2, 4]);
        assert_eq!(mid.data[0], 4.0);
    }

    #[test]
    fn test_mean_3d_matches_old_behavior() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]);
        let m = mean(&x, -1, true);
        assert_eq!(m.shape, vec![1, 2, 1]);
        assert_eq!(m.data, vec![2.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_mean_axis_out_of_range() {
        let _ = mean(&Tensor::zeros(vec![2, 2]), 5, false);
    }

    // ---------- matmul ----------

    #[test]
    fn test_matmul_2d_known_values() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]);
        assert_eq!(matmul(&a, &b).data, vec![4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn test_matmul_parallel_equals_sequential_math() {
        // 20x20 @ 20x20 = 8000 ops, so it takes the blocked parallel path.
        let a = Tensor::new((0..400).map(|i| (i % 7) as f32).collect(), vec![20, 20]);
        let b = Tensor::new((0..400).map(|i| (i % 5) as f32).collect(), vec![20, 20]);
        let got = matmul(&a, &b);
        for i in 0..20 {
            for j in 0..20 {
                let want: f32 = (0..20).map(|l| a.data[i * 20 + l] * b.data[l * 20 + j]).sum();
                assert!((got.data[i * 20 + j] - want).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_matmul_4d_attention_shape() {
        let q = Tensor::new(vec![1.0; 2 * 3 * 4 * 5], vec![2, 3, 4, 5]);
        let k_t = Tensor::new(vec![1.0; 2 * 3 * 5 * 4], vec![2, 3, 5, 4]);
        let scores = matmul(&q, &k_t);
        assert_eq!(scores.shape, vec![2, 3, 4, 4]);
        assert!(scores.data.iter().all(|&x| x == 5.0));
    }

    #[test]
    #[should_panic(expected = "Batch sizes must match")]
    fn test_matmul_4d_batch_mismatch() {
        let a = Tensor::zeros(vec![2, 2, 2, 2]);
        let b = Tensor::zeros(vec![1, 2, 2, 2]);
        let _ = matmul(&a, &b);
    }

    #[test]
    #[should_panic(expected = "Head counts must match")]
    fn test_matmul_4d_head_mismatch() {
        let a = Tensor::zeros(vec![1, 2, 2, 2]);
        let b = Tensor::zeros(vec![1, 3, 2, 2]);
        let _ = matmul(&a, &b);
    }

    // ---------- other ----------

    #[test]
    fn test_reshape_and_concat() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0], vec![1, 2]);
        let c = concat(&a, &b);
        assert_eq!(c.shape, vec![3, 2]);
        assert_eq!(c.reshape(&[6]).data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_var_population() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]);
        assert_close(&var(&x, -1, false).data, &[2.0 / 3.0]);
    }

    #[test]
    fn test_var_axes_and_keepdim() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 6.0], vec![2, 2]);
        // Down the columns: [1, 3] -> 1, [2, 6] -> 4.
        assert_eq!(var(&x, 0, false).data, vec![1.0, 4.0]);
        // Along the rows: [1, 2] -> 0.25, [3, 6] -> 2.25.
        assert_eq!(var(&x, 1, false).data, vec![0.25, 2.25]);
        assert_eq!(var(&x, 0, true).shape, vec![1, 2]);
        assert_eq!(var(&x, -1, true).shape, vec![2, 1]);
    }

    #[test]
    fn test_var_4d_and_middle_axis() {
        // The old code supported only the last axis.
        let x = Tensor::arange(0, 32).reshape(&[2, 2, 2, 4]);
        let last = var(&x, -1, false);
        assert_eq!(last.shape, vec![2, 2, 2]);
        // Every row is 4 consecutive numbers: variance of [0, 1, 2, 3] = 1.25.
        assert!(last.data.iter().all(|&v| (v - 1.25).abs() < 1e-5));
        // Axis 1: pairs (k, k + 8) -> mean k + 4, variance 16.
        let mid = var(&x, 1, false);
        assert_eq!(mid.shape, vec![2, 2, 4]);
        assert!(mid.data.iter().all(|&v| (v - 16.0).abs() < 1e-4));
    }

    #[test]
    fn test_var_matches_mean_of_squared_gaps() {
        // Cross-check against the public operations on a non-last axis.
        let x = Tensor::new(
            (0..24).map(|i| ((i * 5) % 13) as f32).collect(),
            vec![2, 3, 4],
        );
        let gaps = sub(&x, &mean(&x, 1, true));
        let want = mean(&mul(&gaps, &gaps), 1, false);
        assert_close(&var(&x, 1, false).data, &want.data);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_var_axis_out_of_range() {
        let _ = var(&Tensor::zeros(vec![2, 2]), 2, false);
    }

    // ------------------------------------------------------------------
    // Checks that used to live on the slice-level functions
    // ------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "Matrix dimensions incompatible: [2, 3] @ [2, 3]")]
    fn test_matmul_2d_inner_mismatch_panics() {
        let a = Tensor::zeros(vec![2, 3]);
        let b = Tensor::zeros(vec![2, 3]);
        let _ = matmul(&a, &b);
    }

    #[test]
    #[should_panic(expected = "Inner dimensions must match for batched matmul")]
    fn test_matmul_4d_inner_mismatch_panics() {
        let a = Tensor::zeros(vec![1, 2, 2, 3]);
        let b = Tensor::zeros(vec![1, 2, 2, 2]);
        let _ = matmul(&a, &b);
    }

    #[test]
    #[should_panic(expected = "Unsupported matmul shapes: [2, 2, 2] @ [2, 2, 2]")]
    fn test_matmul_unsupported_rank_panics() {
        let a = Tensor::zeros(vec![2, 2, 2]);
        let b = Tensor::zeros(vec![2, 2, 2]);
        let _ = matmul(&a, &b);
    }

    #[test]
    fn test_matmul_4d_values_and_shape() {
        // [batch=1, heads=2, seq=2, dim=3] @ [1, 2, 3, 2], all ones: every number is 3.
        let a = Tensor::new(vec![1.0; 12], vec![1, 2, 2, 3]);
        let b = Tensor::new(vec![1.0; 12], vec![1, 2, 3, 2]);
        let actual = matmul(&a, &b);
        assert_eq!(actual.shape, vec![1, 2, 2, 2]);
        assert!(actual.data.iter().all(|&x| x == 3.0));
    }

    #[test]
    fn test_softmax_empty_tensor() {
        let t = Tensor::zeros(vec![0, 3]);
        let actual = softmax(&t, -1);
        assert_eq!(actual.shape, vec![0, 3]);
        assert!(actual.data.is_empty());
    }

    #[test]
    fn test_masked_fill_same_shape() {
        let scores = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let mask = Tensor::new(vec![0.0, 1.0, 0.0, 0.0], vec![2, 2]);
        let actual = masked_fill(&scores, &mask, f32::NEG_INFINITY);
        let expect = vec![1.0, f32::NEG_INFINITY, 3.0, 4.0];
        assert_eq!(actual.data, expect);
    }

    #[test]
    fn test_masked_fill_reuses_a_2d_mask_for_4d_data() {
        let scores = Tensor::new(vec![1.0; 16], vec![2, 2, 2, 2]);
        let mask = Tensor::new(vec![0.0, 1.0, 0.0, 0.0], vec![2, 2]);
        let actual = masked_fill(&scores, &mask, 0.0);
        let expect: Vec<f32> = (0..16).map(|i| if i % 4 == 1 { 0.0 } else { 1.0 }).collect();
        assert_eq!(actual.data, expect);
    }

    #[test]
    #[should_panic(expected = "Unsupported broadcast for masked_fill")]
    fn test_masked_fill_unsupported_mask_panics() {
        let scores = Tensor::zeros(vec![2, 3]);
        let mask = Tensor::zeros(vec![2, 2]);
        let _ = masked_fill(&scores, &mask, 1.0);
    }

    #[test]
    #[should_panic(expected = "All dimensions except first must match")]
    fn test_concat_mismatched_trailing_dimension_panics() {
        let a = Tensor::zeros(vec![2, 2]);
        let b = Tensor::zeros(vec![1, 3]);
        let _ = concat(&a, &b);
    }

    #[test]
    #[should_panic(expected = "same number of dimensions")]
    fn test_concat_rank_mismatch_panics() {
        let a = Tensor::zeros(vec![2, 2]);
        let b = Tensor::zeros(vec![2]);
        let _ = concat(&a, &b);
    }
}
