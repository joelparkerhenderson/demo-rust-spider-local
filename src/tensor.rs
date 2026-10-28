//! Tensor Operations for Neural Networks
//!
//! This module provides a minimal tensor library optimized for transformer models.
//! Tensors store multi-dimensional arrays with shape and stride information for
//! efficient indexing and memory layout.
//!
//! ## Core Concepts
//!
//! - **Data**: Flat `Vec<f32>` storing all elements in row-major order
//! - **Shape**: Dimensions of the tensor (e.g., `[batch, seq, dim]`)
//! - **Strides**: Step sizes for each dimension to compute flat indices
//!
//! ## Example
//!
//! ```rust
//! use crate::Tensor;
//! use crate::tensor_functions::{matmul};
//!
//! // Create a 2x3 matrix
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let tensor = Tensor::new(data, vec![2, 3]);
//!
//! // Matrix multiplication
//! let other = Tensor::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2]);
//! let result = matmul(&tensor, &other);
//! assert_eq!(result.shape, vec![2, 2]);
//! ```
//!
//! ## Performance Optimizations
//!
//! Several operations use parallel processing via Rayon for performance:
//!
//! - **Matrix multiplication**: Cache-blocked algorithm with parallel row processing
//! - **Element-wise operations**: Parallel iteration over data
//! - **Softmax**: Parallel computation per row
//!
//! These optimizations provide 2-4x speedup on multi-core CPUs while keeping
//! the code educational and understandable. All optimizations are clearly marked
//! in the code with comments explaining the approach.
//!
//! # A guided tour (tutorial)
//!
//! This file is written to be read top to bottom. Suggested path:
//!
//! 1. This header: the mental model (below).
//! 2. [`Tensor`]: the struct, and its three invariants.
//! 3. `new`, `zeros`: how a tensor is built and checked (they use the
//!    `compute_strides` helper).
//! 4. The scalar operations (`add_scalar`, `mul_scalar`, `div_scalar`, `sqrt`),
//!    `reshape` and `arange`: the few operations that live on `Tensor` itself.
//! 5. `tensor_functions.rs`: ALL the other operations, as pure functions that
//!    take tensors and return a new tensor: `matmul` (the most important
//!    one), `softmax`, `add`/`sub`/`mul`/`div` (with broadcasting), `transpose`,
//!    `masked_fill`, `mean`, `var` and `concat` (the pieces a transformer needs:
//!    masking, layer normalization, KV caching), plus the helpers beneath them.
//!
//! ## The mental model: a flat list read through a shape
//!
//! Memory is one long line. A tensor is a flat line of numbers plus a
//! `shape` that tells us how to cut the line into rows, blocks and so on.
//!
//! ```text
//! shape [2, 3]:   data = [ a  b  c  d  e  f ]
//!
//!                 read as:   a b c        <- row 0
//!                            d e f        <- row 1
//!
//! element [row, col]  lives at  data[row * 3 + col]
//!                                      ^^^^^^^ stride of dim 0 (= number of columns)
//! ```
//!
//! For shape `[d0, d1, d2]` the strides are `[d1*d2, d2, 1]`, and element
//! `[i, j, k]` lives at `data[i*d1*d2 + j*d2 + k]`. Nearly every loop in this
//! file is this one formula written out by hand.
//!
//! ## What the shapes mean in a transformer
//!
//! | Shape                          | Meaning                                  |
//! |--------------------------------|------------------------------------------|
//! | `[vocab, dim]`                 | embedding table, one row per token       |
//! | `[seq, dim]`                   | one sentence: a vector per token         |
//! | `[batch, seq, dim]`            | several sentences at once                |
//! | `[batch, heads, seq, head_dim]`| queries/keys/values split into heads     |
//! | `[batch, heads, seq, seq]`     | attention scores (every token vs. every token) |
//!
//! ## How operations are chosen: shape dispatch
//!
//! This is a deliberately small library. Each operation checks the shapes
//! it was given and handles only the cases a transformer needs (for example,
//! `matmul` supports 2-D and 4-D, not 3-D). Anything else ends in a
//! `panic!` with a message that prints both shapes. When you see
//! "Unsupported ... shapes", the fix is usually to reshape first.
//!
//! ## Two ideas used throughout
//!
//! - **Out-of-place**: every operation returns a NEW `Tensor` and leaves its
//!   inputs untouched. It is easy to reason about; the price is an
//!   allocation per call.
//! - **Pure functions**: the operations are free functions (in
//!   `tensor_functions.rs`) that take all their inputs as arguments (no `self`):
//!   `matmul(&a, &b)`, `softmax(&t, -1)`. Each can be read and tested in
//!   isolation. A method is kept only where it works directly on the tensor's
//!   own fields (construction, `reshape`, the scalar operations).
//! - **Panics, not `Result`**: shape mistakes are programmer errors, so they
//!   panic with a clear message, much like indexing a slice out of range.
//!
//! ## Known limitations (read before relying on an operation)
//!
//! - `matmul` supports only 2-D and 4-D inputs (no 3-D).
//! - `masked_fill` supports identical shapes, or a 4-D tensor with a 2-D mask.
//! - Broadcasting in `add`/`sub`/`mul`/`div` decodes each output position's
//!   coordinates, which is simple and general but slower than a hand-written
//!   special case such as `i % n` for a bias vector.
//! - `reshape` and `transpose` copy their data (no shared views).
//!
//! Behaviors worth knowing, which are by design:
//!
//! - A `softmax` group that is entirely `-inf` returns zeros (not `NaN`).
//! - The mean of an empty axis is `NaN`; division by zero gives `inf`/`NaN`.
//! - Shape or axis mistakes panic with a message that prints the shapes.

use crate::tensor_functions::compute_strides;
use rayon::prelude::*;

/// A multi-dimensional array for neural network computations
///
/// Tensors store data in a contiguous `Vec<f32>` with shape and stride information
/// for efficient multi-dimensional indexing. All operations use row-major (C-style)
/// memory layout.
///
/// # Fields
///
/// - `data`: Flat array of f32 values
/// - `shape`: Dimensions (e.g., `[2, 3]` for a 2x3 matrix)
/// - `strides`: Step sizes for each dimension (computed from shape)
///
/// # Memory Layout
///
/// For shape `[2, 3]`, data is stored as: `[row0_col0, row0_col1, row0_col2, row1_col0, row1_col1, row1_col2]`
///
/// Strides would be `[3, 1]` meaning:
/// - Moving one step in dimension 0 (rows) advances 3 positions in data
/// - Moving one step in dimension 1 (cols) advances 1 position in data
///
/// # Invariants
///
/// [`Tensor::new`] enforces the first rule and computes the second, so any
/// tensor made through the public constructors satisfies both:
///
/// 1. `data.len() == shape.iter().product()` (one number per grid cell).
/// 2. `strides` are the row-major strides of `shape`.
///
/// The fields are `pub`, so code that edits them directly (for example,
/// pushing onto `data`) can break these rules. Prefer building a new
/// tensor with `Tensor::new`.
///
/// # Derives
///
/// - `Clone` makes a deep copy of all three vectors. It can be expensive
///   for large tensors; this file calls it deliberately in `reshape`.
/// - `Debug` allows `println!("{:?}", tensor)` for inspection.
#[derive(Clone, Debug)]
pub struct Tensor {
    /// Flat storage of all tensor elements
    ///
    /// Row-major: the LAST dimension changes fastest. `f32` is the usual
    /// precision for neural networks (half the memory of `f64`).
    pub data: Vec<f32>,
    /// Shape of the tensor (dimensions)
    ///
    /// Outermost dimension first. An empty shape `[]` is a scalar, and the
    /// product of an empty list is 1, so a scalar holds one number.
    pub shape: Vec<usize>,
    /// Strides for each dimension (computed from shape)
    ///
    /// `strides[i]` is how many positions in `data` to skip to move one
    /// step along dimension `i`.
    pub strides: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor with given data and shape
    ///
    /// This is the one constructor every other function in this file ends
    /// up calling, so it is where the data-length invariant is enforced.
    ///
    /// # Arguments
    ///
    /// * `data` - Flat vector of values
    /// * `shape` - Dimensions of the tensor
    ///
    /// # Panics
    ///
    /// Panics if the product of shape dimensions doesn't equal data length
    ///
    /// # Example
    ///
    /// ```rust
    /// use crate::Tensor;
    /// let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    /// assert_eq!(tensor.shape, vec![2, 2]);
    /// ```
    ///
    /// A mismatch panics, which is how shape bugs are caught early:
    ///
    /// ```rust,should_panic
    /// use crate::Tensor;
    /// // 5 numbers cannot fill a 2 x 3 grid (needs 6).
    /// let _ = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]);
    /// ```
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        // Step 1: how many numbers does this shape need?
        // `product()` multiplies every dimension: [2, 3, 4] -> 24.
        // For an empty shape (a scalar) the product is 1.
        let expected_size: usize = shape.iter().product();

        // Step 2: refuse to build an inconsistent tensor. Failing here, at
        // the moment of construction, is much easier to debug than a wrong
        // answer or an out-of-range index many operations later.
        assert_eq!(
            data.len(),
            expected_size,
            "Data length ({}) doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_size
        );

        // Step 3: derive the strides. They are always computed, never passed
        // in, so they can never disagree with the shape.
        let strides = compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }

    /// Create a tensor filled with zeros
    ///
    /// Zeros are the usual starting value for an accumulator.
    ///
    /// # Arguments
    ///
    /// * `shape` - Dimensions of the tensor
    ///
    /// # Example
    ///
    /// ```rust
    /// use crate::Tensor;
    /// let tensor = Tensor::zeros(vec![3, 4]);
    /// assert_eq!(tensor.data.len(), 12);
    /// assert!(tensor.data.iter().all(|&x| x == 0.0));
    /// ```
    pub fn zeros(shape: Vec<usize>) -> Self {
        // How many cells? (3 * 4 = 12 for the example above.)
        let size: usize = shape.iter().product();
        // `vec![0.0; size]` means "`size` copies of 0.0".
        let data = vec![0.0; size];
        // Reuse `new` so the length check and strides are computed in one place.
        Self::new(data, shape)
    }

    /// Add scalar to all elements
    ///
    /// A "scalar" here is a plain `f32`, not a rank-0 tensor. Shape is kept.
    pub fn add_scalar(&self, scalar: f32) -> Tensor {
        // `|&x|` destructures the `&f32` that `par_iter` yields, so `x` is
        // a plain `f32`. The closure captures `scalar` by reference; that is
        // safe across threads because `f32` is `Copy` and read-only.
        let result = self.data.par_iter().map(|&x| x + scalar).collect();
        Tensor::new(result, self.shape.clone())
    }

    /// Multiply all elements by scalar
    ///
    /// Used, for example, to scale attention scores by `1 / sqrt(head_dim)`.
    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        let result = self.data.par_iter().map(|&x| x * scalar).collect();
        Tensor::new(result, self.shape.clone())
    }

    /// Divide all elements by scalar
    ///
    /// Dividing by `0.0` gives `inf`/`NaN` rather than a panic.
    pub fn div_scalar(&self, scalar: f32) -> Tensor {
        let result = self.data.par_iter().map(|&x| x / scalar).collect();
        Tensor::new(result, self.shape.clone())
    }

    /// Element-wise square root
    ///
    /// The square root of a negative number is `NaN` (not a number); no
    /// panic occurs. `NaN` then spreads through any later arithmetic.
    pub fn sqrt(&self) -> Tensor {
        let result = self.data.par_iter().map(|&x| x.sqrt()).collect();
        Tensor::new(result, self.shape.clone())
    }

    /// Reshape tensor to new shape
    ///
    /// Total number of elements must remain the same.
    ///
    /// # Example
    ///
    /// ```rust
    /// use crate::Tensor;
    /// let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    /// let reshaped = tensor.reshape(&[3, 2]);
    /// assert_eq!(reshaped.shape, vec![3, 2]);
    /// ```
    ///
    /// # What reshape does (and does not do)
    ///
    /// The numbers keep their order; only the way they are cut into rows
    /// changes.
    ///
    /// ```text
    /// [2, 3]            [3, 2]            [6]
    /// 1 2 3             1 2               1 2 3 4 5 6
    /// 4 5 6             3 4
    ///                   5 6
    /// data (all three):  [1, 2, 3, 4, 5, 6]
    /// ```
    ///
    /// Compare with [`transpose`](crate::tensor_functions::transpose), which REORDERS the numbers.
    ///
    /// This version copies the data (`clone`) rather than sharing it. A
    /// production library would reuse the buffer and change only the shape,
    /// which is why reshape is "free" in PyTorch. Copying keeps this code
    /// simple, and each tensor owns its data, at the cost of a copy.
    ///
    /// # Panics
    ///
    /// Panics if the new shape holds a different number of elements.
    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        // The product of the new shape must equal the current element count.
        let new_size: usize = new_shape.iter().product();
        assert_eq!(
            self.data.len(),
            new_size,
            "Cannot reshape: element count mismatch"
        );
        // `new_shape.to_vec()` copies the borrowed slice into an owned Vec.
        // `new` recomputes the strides for the new shape.
        Tensor::new(self.data.clone(), new_shape.to_vec())
    }

    /// Create a tensor with sequential integers
    ///
    /// # Arguments
    ///
    /// * `start` - Starting value (inclusive)
    /// * `end` - Ending value (exclusive)
    ///
    /// # Example
    ///
    /// ```rust
    /// use crate::Tensor;
    /// let tensor = Tensor::arange(0, 5);
    /// assert_eq!(tensor.data, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    /// ```
    ///
    /// Works like Python's `range`: the start is included and the end is
    /// not. The result is 1-D with shape `[end - start]`, and its values are
    /// `f32`. Both bounds are `usize`, so negative starts are not possible;
    /// if `end <= start` the result is empty (shape `[0]`).
    ///
    /// Handy with [`Tensor::reshape`] for making test grids:
    /// `Tensor::arange(0, 6).reshape(&[2, 3])`.
    pub fn arange(start: usize, end: usize) -> Tensor {
        // `(start..end)` is an integer range; `map` converts each to f32.
        let data: Vec<f32> = (start..end).map(|i| i as f32).collect();
        // The length is whatever the range produced.
        let len = data.len();
        Tensor::new(data, vec![len])
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- construction ----------

    #[test]
    fn test_strides() {
        let t = Tensor::zeros(vec![2, 3, 4]);
        assert_eq!(t.strides, vec![12, 4, 1]);
        assert_eq!(Tensor::zeros(vec![5]).strides, vec![1]);
        assert!(Tensor::zeros(vec![]).strides.is_empty());
    }

    #[test]
    #[should_panic(expected = "doesn't match shape")]
    fn test_new_rejects_wrong_length() {
        let _ = Tensor::new(vec![1.0; 5], vec![2, 3]);
    }
}
