//! Pure functions for the AdamW optimizer
//!
//! Project convention (see `spec/index.md`): prefer pure functions, and make
//! functions public by default. This file holds the arithmetic of one AdamW
//! step for a single tensor, with no knowledge of the model:
//!
//! - [`StepConstants`]: the numbers shared by every parameter in one step
//!   (learning rate, β's, ε, and the bias-correction factors).
//! - [`adamw_update_tensor`]: update ONE parameter tensor from its gradient and
//!   its two moment tensors.
//! - [`zeros_like`]: a tensor of zeros shaped like a parameter (the starting
//!   value of its moment tensors).
//!
//! `optimizer.rs` holds the optimizer's state and applies these to every
//! parameter of the GPT-2 model. Read its module docs first for the tutorial:
//! what momentum, the second moment, bias correction and decoupled weight decay
//! are, with worked numbers. The unit tests for those numbers are at the bottom
//! of this file.
//!
//! ```text
//! θ = θ * (1 - lr * λ)              # weight decay (if applicable)
//! m = β₁ * m + (1 - β₁) * g         # first moment (momentum)
//! v = β₂ * v + (1 - β₂) * g²        # second moment (variance)
//! θ = θ - lr * (m / (1 - β₁^t)) / (√(v / (1 - β₂^t)) + ε)
//! ```

use crate::tensor::Tensor;
use rayon::prelude::*;

/// Settings that are the same for every parameter within one optimizer step
///
/// Bundling them keeps `adamw_update_tensor` from taking a long list of
/// separate numbers, and makes sure the bias-correction factors are computed
/// once per step (they depend only on the step count), not once per tensor.
#[derive(Clone, Copy)]
pub struct StepConstants {
    /// Learning rate (α)
    pub lr: f32,
    /// Weight-decay coefficient (λ)
    pub weight_decay: f32,
    /// Momentum decay rate (β₁)
    pub beta1: f32,
    /// Variance decay rate (β₂)
    pub beta2: f32,
    /// Numerical-stability constant (ε)
    pub epsilon: f32,
    /// `1 - β₁^t`: scales the first moment back up
    pub bias_correction1: f32,
    /// `1 - β₂^t`: scales the second moment back up
    pub bias_correction2: f32,
}

impl StepConstants {
    /// Compute the constants for step number `step` (the first step is 1)
    ///
    /// # Bias-correction factors
    ///
    /// These correct for initialization bias (m and v start at 0).
    /// At t = 1: `1 - 0.9 = 0.1` and `1 - 0.95 = 0.05`, so m and v are scaled
    /// up by 10x and 20x. As t grows, β^t -> 0 and the factors approach 1.
    pub fn new(
        lr: f32,
        weight_decay: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        step: usize,
    ) -> Self {
        let step = step as f32;
        Self {
            lr,
            weight_decay,
            beta1,
            beta2,
            epsilon,
            bias_correction1: 1.0 - beta1.powf(step),
            bias_correction2: 1.0 - beta2.powf(step),
        }
    }
}

/// A tensor of zeros with the same shape as `tensor`
///
/// The optimizer's moment tensors (`m` and `v`) start as zeros and mirror the
/// shape of the parameter they belong to. Only the SHAPE of `tensor` is read,
/// never its values.
///
/// ```text
/// weights [2, 3]: 0.5 -1.2 0.3     ->  zeros_like: 0 0 0
///                 0.9  0.0 2.1                     0 0 0
/// ```
///
/// # Arguments
///
/// * `tensor` - the tensor whose shape to copy
///
/// # Returns
///
/// A new tensor, same shape, every number 0.0.
pub fn zeros_like(tensor: &Tensor) -> Tensor {
    Tensor::zeros(tensor.shape.clone())
}

/// Apply one AdamW step to a single parameter tensor
///
/// This is the heart of the optimizer, written once and called for every
/// parameter. (An earlier version used a `macro_rules!` for this; a plain
/// function is easier to read, to test, and to get clear error messages from.)
///
/// # Arguments
///
/// * `name` - Parameter name, used only in error messages
/// * `param` - The weights to change (modified in place)
/// * `grad` - The gradient for `param` (same shape)
/// * `m` - First moment for `param` (updated in place, same shape)
/// * `v` - Second moment for `param` (updated in place, same shape)
/// * `c` - The step's constants (learning rate, β's, bias corrections ...)
/// * `apply_decay` - Whether to shrink the weights by `1 - lr * weight_decay`
///
/// # Panics
///
/// Panics if `grad`, `m` or `v` does not have the same shape as `param`. The
/// four tensors are walked in lock-step, so a mismatch would otherwise either
/// panic with a confusing index error (small tensors) or silently update only a
/// prefix of the weights (large tensors, where `zip` stops at the shortest).
///
/// Also panics if `grad` contains `NaN` or infinity (see the comment at the
/// check). The panic happens before this tensor is touched, but tensors earlier
/// in the same step have already been updated, so after such a panic the
/// model and optimizer should be discarded or reloaded from a checkpoint.
///
/// # Performance
///
/// Tensors with more than 1000 elements are updated in parallel with Rayon,
/// because every element is independent. Smaller tensors use a plain loop,
/// because starting threads costs more than the work.
pub fn adamw_update_tensor(
    name: &str,
    param: &mut Tensor,
    grad: &Tensor,
    m: &mut Tensor,
    v: &mut Tensor,
    c: &StepConstants,
    apply_decay: bool,
) {
    // Check shapes first: the loops below trust that all four tensors line up.
    assert!(
        param.shape == grad.shape,
        "adamw_update: gradient shape {:?} does not match parameter `{}` shape {:?}",
        grad.shape,
        name,
        param.shape
    );
    assert!(
        param.shape == m.shape && param.shape == v.shape,
        "adamw_update: optimizer state shapes {:?} / {:?} do not match parameter `{}` shape {:?} \
         (was the optimizer created for a different model?)",
        m.shape,
        v.shape,
        name,
        param.shape
    );

    // Refuse a gradient containing NaN or infinity.
    //
    // WHY: `m` and `v` are running averages that REMEMBER every gradient. One
    // NaN would be mixed into them, then into the weights, and every later step
    // would produce NaN: training is silently ruined, and the only symptom is a
    // loss of NaN many steps later. Failing here points straight at the
    // parameter. `par_iter().any(..)` scans in parallel; the slower search for
    // the exact position runs only when something is wrong.
    if grad.data.par_iter().any(|x| !x.is_finite()) {
        let index = grad.data.iter().position(|x| !x.is_finite()).unwrap_or(0);
        panic!(
            "adamw_update: gradient for `{}` contains a non-finite value ({}) at index {}; \
             the optimizer state was not changed for this tensor",
            name, grad.data[index], index
        );
    }

    // Unpack the constants into plain local variables so the closure below
    // captures simple copies.
    let StepConstants {
        lr,
        weight_decay,
        beta1,
        beta2,
        epsilon,
        bias_correction1,
        bias_correction2,
    } = *c;

    // Parallelize for large tensors (>1000 elements)
    //
    // Every element is updated independently of all the others, so
    // the work can be split across CPU cores with no coordination.
    if param.data.len() > 1000 {
        // `par_iter_mut` hands out `&mut` to the parameters;
        // `par_iter` hands out `&` to the gradients. Two `zip`s line
        // up the four arrays so each step sees one element of each:
        //   ((param, grad), (m, v))
        // The closure pattern below unpacks exactly that nesting;
        // `&grad_val` copies the gradient number out of its reference.
        param
            .data
            .par_iter_mut()
            .zip(grad.data.par_iter())
            .zip(m.data.par_iter_mut().zip(v.data.par_iter_mut()))
            .for_each(|((param_val, &grad_val), (m_val, v_val))| {
                // WEIGHT DECAY: Apply before Adam update (decoupled)
                //
                // Shrinks the weight by the factor (1 - lr*λ), e.g.
                // 0.99997. This uses the weight BEFORE the gradient
                // step, as in the AdamW paper. The gradient and the
                // moments below do not see the decay at all: that is
                // what "decoupled" means.
                if apply_decay {
                    *param_val *= 1.0 - lr * weight_decay;
                }

                // Update biased first moment estimate (momentum)
                //
                // New m = 90% old m + 10% new gradient.
                // (`*m_val` reads through the `&mut` reference.)
                *m_val = beta1 * *m_val + (1.0 - beta1) * grad_val;

                // Update biased second moment estimate (variance)
                //
                // New v = 95% old v + 5% squared new gradient.
                // Squaring makes it always >= 0 and tracks magnitude.
                *v_val = beta2 * *v_val + (1.0 - beta2) * grad_val * grad_val;

                // Compute bias-corrected first moment
                //
                // Undo the pull toward zero from the zero start.
                let m_hat = *m_val / bias_correction1;

                // Compute bias-corrected second moment
                let v_hat = *v_val / bias_correction2;

                // Update parameter (Adam step)
                //
                // m_hat gives the DIRECTION (smoothed gradient);
                // dividing by √v_hat (typical gradient size) makes the
                // step about `lr` in size for every parameter.
                // `epsilon` is added AFTER the square root so a tiny
                // or zero v_hat cannot cause division by zero.
                // Minus sign: move AGAINST the gradient to reduce loss.
                *param_val -= lr * m_hat / (v_hat.sqrt() + epsilon);
            });
    } else {
        // Sequential for small tensors to avoid parallelization overhead
        //
        // Same arithmetic, in a plain loop. Starting threads costs more
        // than the work for a few hundred numbers (biases, LayerNorm).
        for i in 0..param.data.len() {
            // WEIGHT DECAY: Apply before Adam update (decoupled)
            if apply_decay {
                param.data[i] *= 1.0 - lr * weight_decay;
            }

            // The gradient for this element.
            let g = grad.data[i];

            // Update biased first moment estimate (momentum)
            m.data[i] = beta1 * m.data[i] + (1.0 - beta1) * g;

            // Update biased second moment estimate (variance)
            v.data[i] = beta2 * v.data[i] + (1.0 - beta2) * g * g;

            // Compute bias-corrected first moment
            let m_hat = m.data[i] / bias_correction1;

            // Compute bias-corrected second moment
            let v_hat = v.data[i] / bias_correction2;

            // Update parameter (Adam step)
            param.data[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the per-tensor update, which is where all the math is.
    //! They need only `Tensor`, not a full model. Expected numbers are the
    //! hand-worked examples from the module docs.
    use super::*;

    /// Constants for step `step` with the usual transformer settings.
    fn constants(step: usize, lr: f32, weight_decay: f32) -> StepConstants {
        StepConstants::new(lr, weight_decay, 0.9, 0.95, 1e-8, step)
    }

    /// One-number tensors, for easy hand checking.
    fn scalar(x: f32) -> Tensor {
        Tensor::new(vec![x], vec![1])
    }

    /// Run one update on a single number; returns (param, m, v).
    fn one_step(theta: f32, g: f32, c: &StepConstants, decay: bool, m0: f32, v0: f32) -> (f32, f32, f32) {
        let mut p = scalar(theta);
        let mut m = scalar(m0);
        let mut v = scalar(v0);
        adamw_update_tensor("p", &mut p, &scalar(g), &mut m, &mut v, c, decay);
        (p.data[0], m.data[0], v.data[0])
    }

    #[test]
    fn test_bias_correction_factors() {
        let c = constants(1, 1.0, 0.0);
        assert!((c.bias_correction1 - 0.1).abs() < 1e-6);
        assert!((c.bias_correction2 - 0.05).abs() < 1e-6);
        // They approach 1 as t grows.
        let late = constants(1000, 1.0, 0.0);
        assert!((late.bias_correction1 - 1.0).abs() < 1e-6);
        assert!((late.bias_correction2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_first_step_worked_example_with_decay() {
        // The example in the docs: θ=1, g=0.5, lr=3e-4, λ=0.1 -> 0.99967.
        let c = constants(1, 3e-4, 0.1);
        let (theta, m, v) = one_step(1.0, 0.5, &c, true, 0.0, 0.0);
        assert!((m - 0.05).abs() < 1e-7, "m = {}", m);
        assert!((v - 0.0125).abs() < 1e-7, "v = {}", v);
        assert!((theta - 0.99967).abs() < 1e-6, "theta = {}", theta);
    }

    #[test]
    fn test_first_step_without_decay() {
        let c = constants(1, 3e-4, 0.1);
        let (theta, _, _) = one_step(1.0, 0.5, &c, false, 0.0, 0.0);
        // Only the Adam step (about lr): 1 - 0.0003.
        assert!((theta - 0.9997).abs() < 1e-6, "theta = {}", theta);
    }

    #[test]
    fn test_decay_does_not_change_moments() {
        // "Decoupled": the moments must be identical with and without decay.
        let c = constants(3, 1e-3, 0.1);
        let a = one_step(2.0, 0.7, &c, true, 0.2, 0.3);
        let b = one_step(2.0, 0.7, &c, false, 0.2, 0.3);
        assert_eq!(a.1, b.1);
        assert_eq!(a.2, b.2);
        assert_ne!(a.0, b.0);
    }

    #[test]
    fn test_zero_gradient_only_decays() {
        // g = 0 with empty moments: no Adam step, just the shrink factor.
        let c = constants(1, 3e-4, 0.1);
        let (theta, m, v) = one_step(2.0, 0.0, &c, true, 0.0, 0.0);
        assert_eq!((m, v), (0.0, 0.0));
        assert!((theta - 2.0 * 0.99997).abs() < 1e-6);
        // And with decay off, nothing changes at all.
        let (theta, _, _) = one_step(2.0, 0.0, &c, false, 0.0, 0.0);
        assert_eq!(theta, 2.0);
    }

    #[test]
    fn test_constant_gradient_moves_about_lr_each_step() {
        // With bias correction, a steady gradient gives a step of ~lr each time,
        // whatever the gradient's size.
        for g in [0.5_f32, 50.0, 0.001] {
            let (mut theta, mut m, mut v) = (0.0_f32, 0.0_f32, 0.0_f32);
            for t in 1..=5 {
                let c = constants(t, 1e-3, 0.0);
                let before = theta;
                let r = one_step(theta, g, &c, false, m, v);
                theta = r.0;
                m = r.1;
                v = r.2;
                assert!(((before - theta) - 1e-3).abs() < 1e-5, "g {} step {}: moved {}", g, t, before - theta);
            }
        }
    }

    #[test]
    fn test_alternating_gradient_matches_doc_table() {
        // Docs: gradients +1, -1, +1, -1 with lr = 1 give steps
        // +1.000, -0.053, +0.336, -0.053 (a step is subtracted from theta).
        let want_steps = [1.0_f32, -0.05263, 0.33579, -0.05263];
        let (mut theta, mut m, mut v) = (0.0_f32, 0.0_f32, 0.0_f32);
        for (i, g) in [1.0_f32, -1.0, 1.0, -1.0].into_iter().enumerate() {
            let c = constants(i + 1, 1.0, 0.0);
            let before = theta;
            let r = one_step(theta, g, &c, false, m, v);
            theta = r.0;
            m = r.1;
            v = r.2;
            assert!(((before - theta) - want_steps[i]).abs() < 1e-4, "step {}: {}", i + 1, before - theta);
        }
    }

    #[test]
    fn test_parallel_path_matches_sequential_path() {
        // 2000 elements takes the Rayon path; one element takes the loop.
        // Same inputs must give bit-identical results.
        let n = 2000;
        let thetas: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let grads: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).cos() * 0.1).collect();
        let c = constants(4, 3e-4, 0.1);

        let mut p = Tensor::new(thetas.clone(), vec![n]);
        let mut m = Tensor::zeros(vec![n]);
        let mut v = Tensor::zeros(vec![n]);
        adamw_update_tensor("big", &mut p, &Tensor::new(grads.clone(), vec![n]), &mut m, &mut v, &c, true);

        for i in 0..n {
            let (theta, mi, vi) = one_step(thetas[i], grads[i], &c, true, 0.0, 0.0);
            assert_eq!(p.data[i], theta, "param {}", i);
            assert_eq!(m.data[i], mi, "m {}", i);
            assert_eq!(v.data[i], vi, "v {}", i);
        }
    }

    #[test]
    fn test_works_on_2d_tensor() {
        // Shape does not matter to the update, only matching shapes.
        let mut p = Tensor::new(vec![1.0; 6], vec![2, 3]);
        let mut m = Tensor::zeros(vec![2, 3]);
        let mut v = Tensor::zeros(vec![2, 3]);
        let g = Tensor::new(vec![0.5; 6], vec![2, 3]);
        adamw_update_tensor("w", &mut p, &g, &mut m, &mut v, &constants(1, 3e-4, 0.1), true);
        assert!(p.data.iter().all(|&x| (x - 0.99967).abs() < 1e-6));
        assert_eq!(p.shape, vec![2, 3]);
    }

    #[test]
    #[should_panic(expected = "does not match parameter `w`")]
    fn test_gradient_shape_mismatch_panics() {
        let mut p = Tensor::zeros(vec![2, 3]);
        let mut m = Tensor::zeros(vec![2, 3]);
        let mut v = Tensor::zeros(vec![2, 3]);
        let g = Tensor::zeros(vec![3, 2]);
        adamw_update_tensor("w", &mut p, &g, &mut m, &mut v, &constants(1, 1.0, 0.0), false);
    }

    #[test]
    #[should_panic(expected = "optimizer state shapes")]
    fn test_state_shape_mismatch_panics() {
        let mut p = Tensor::zeros(vec![4]);
        let mut m = Tensor::zeros(vec![5]);
        let mut v = Tensor::zeros(vec![4]);
        let g = Tensor::zeros(vec![4]);
        adamw_update_tensor("w", &mut p, &g, &mut m, &mut v, &constants(1, 1.0, 0.0), false);
    }

    #[test]
    #[should_panic(expected = "does not match parameter")]
    fn test_large_tensor_shape_mismatch_panics_instead_of_updating_a_prefix() {
        // Above 1000 elements the old code would silently update only a prefix.
        let mut p = Tensor::zeros(vec![2000]);
        let mut m = Tensor::zeros(vec![2000]);
        let mut v = Tensor::zeros(vec![2000]);
        let g = Tensor::zeros(vec![1500]);
        adamw_update_tensor("big", &mut p, &g, &mut m, &mut v, &constants(1, 1.0, 0.0), false);
    }

    #[test]
    #[should_panic(expected = "gradient for `w` contains a non-finite value (NaN) at index 2")]
    fn test_nan_gradient_panics_with_position() {
        let mut p = Tensor::new(vec![1.0; 4], vec![4]);
        let mut m = Tensor::zeros(vec![4]);
        let mut v = Tensor::zeros(vec![4]);
        let g = Tensor::new(vec![0.1, 0.2, f32::NAN, 0.4], vec![4]);
        adamw_update_tensor("w", &mut p, &g, &mut m, &mut v, &constants(1, 1e-3, 0.1), true);
    }

    #[test]
    #[should_panic(expected = "non-finite value (inf)")]
    fn test_infinite_gradient_panics() {
        let mut p = Tensor::zeros(vec![2]);
        let mut m = Tensor::zeros(vec![2]);
        let mut v = Tensor::zeros(vec![2]);
        let g = Tensor::new(vec![f32::INFINITY, 0.0], vec![2]);
        adamw_update_tensor("w", &mut p, &g, &mut m, &mut v, &constants(1, 1e-3, 0.1), false);
    }

    #[test]
    #[should_panic(expected = "non-finite value (NaN) at index 1999")]
    fn test_nan_in_large_tensor_is_found() {
        // The parallel path (more than 1000 elements) must catch it too.
        let mut data = vec![0.1_f32; 2000];
        data[1999] = f32::NAN;
        let mut p = Tensor::zeros(vec![2000]);
        let mut m = Tensor::zeros(vec![2000]);
        let mut v = Tensor::zeros(vec![2000]);
        let g = Tensor::new(data, vec![2000]);
        adamw_update_tensor("big", &mut p, &g, &mut m, &mut v, &constants(1, 1e-3, 0.1), false);
    }

    #[test]
    fn test_nan_gradient_leaves_this_tensor_untouched() {
        let mut p = Tensor::new(vec![1.0, 2.0], vec![2]);
        let mut m = Tensor::new(vec![0.3, 0.4], vec![2]);
        let mut v = Tensor::new(vec![0.5, 0.6], vec![2]);
        let g = Tensor::new(vec![f32::NAN, 1.0], vec![2]);
        let c = constants(1, 1e-3, 0.1);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            adamw_update_tensor("w", &mut p, &g, &mut m, &mut v, &c, true);
        }));
        assert!(result.is_err());
        assert_eq!(p.data, vec![1.0, 2.0]);
        assert_eq!(m.data, vec![0.3, 0.4]);
        assert_eq!(v.data, vec![0.5, 0.6]);
    }

    #[test]
    fn test_large_finite_gradients_are_accepted() {
        // Big but finite values are legitimate (they may just be clipped later).
        let mut p = Tensor::zeros(vec![3]);
        let mut m = Tensor::zeros(vec![3]);
        let mut v = Tensor::zeros(vec![3]);
        let g = Tensor::new(vec![1e10, -1e10, 0.0], vec![3]);
        adamw_update_tensor("w", &mut p, &g, &mut m, &mut v, &constants(1, 1e-3, 0.0), false);
        assert!(p.data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_zeros_like_copies_the_shape_not_the_values() {
        let weights = Tensor::new(vec![0.5, -1.2, 0.3, 0.9, 0.0, 2.1], vec![2, 3]);
        let actual = zeros_like(&weights);
        assert_eq!(actual.shape, vec![2, 3]);
        assert_eq!(actual.data, vec![0.0; 6]);
        // The strides come out right too, because they are computed from the shape.
        assert_eq!(actual.strides, weights.strides);
    }

    #[test]
    fn test_zeros_like_scalar_and_vector() {
        assert_eq!(zeros_like(&Tensor::new(vec![7.0], vec![])).data, vec![0.0]);
        assert_eq!(zeros_like(&Tensor::new(vec![1.0, 2.0, 3.0], vec![3])).shape, vec![3]);
    }
}
