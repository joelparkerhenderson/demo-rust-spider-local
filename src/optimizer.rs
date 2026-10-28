//! AdamW Optimizer Implementation
//!
//! <https://github.com/tag1consulting/feste/blob/main/src/optimizer.rs>
//!
//! This module implements the AdamW (Adam with decoupled Weight decay) optimizer,
//! the standard optimizer for training transformer models like GPT-2.
//!
//! ## What is AdamW?
//!
//! AdamW improves upon Adam by decoupling weight decay from the gradient-based
//! optimization. It combines:
//! - **Momentum**: Smooths gradient updates using exponential moving average
//! - **RMSProp**: Adapts learning rate per-parameter based on gradient history
//! - **Decoupled weight decay**: L2 regularization applied directly to weights
//!
//! ## Algorithm
//!
//! For each parameter θ with gradient g:
//!
//! ```text
//! # AdamW update:
//! θ = θ * (1 - α * λ)              # Weight decay (if applicable)
//! m = β₁ * m + (1 - β₁) * g        # First moment (momentum)
//! v = β₂ * v + (1 - β₂) * g²       # Second moment (variance)
//! m_hat = m / (1 - β₁^t)           # Bias correction
//! v_hat = v / (1 - β₂^t)           # Bias correction
//! θ = θ - α * m_hat / (√v_hat + ε) # Parameter update
//! ```
//!
//! where:
//! - α (alpha/lr) = learning rate (typically 3e-4 for transformers)
//! - λ (lambda/weight_decay) = 0.1 (typical for transformers)
//! - β₁ (beta1) = 0.9 (momentum decay rate)
//! - β₂ (beta2) = 0.95 (variance decay rate, lower than Adam's 0.999)
//! - ε (epsilon) = 1e-8 (numerical stability)
//! - t = training step number
//!
//! ## Why AdamW over Adam?
//!
//! **vs Adam**:
//! - Better generalization: the weight decay is applied directly to the weights,
//!   separate from the gradient (not identical to an L2 penalty under Adam)
//! - Weight decay doesn't interact with adaptive learning rates
//! - Standard choice for modern transformer training (GPT, BERT, etc.)
//!
//! **vs SGD**:
//! - Faster convergence (fewer training steps)
//! - Less sensitive to learning rate choice
//! - Adaptive per-parameter learning rates
//!
//! ## Selective Weight Decay
//!
//! Following best practices from modern transformer training, weight decay is
//! applied **only to 2D tensors (weight matrices)**, not to:
//! - Biases (1D tensors)
//! - LayerNorm parameters (1D tensors)
//! - Embeddings (off by default; `adamw_update_with_options` can turn it on)
//!
//! This selective application prevents over-regularization of scale/shift
//! parameters while still providing L2 regularization benefits for weight matrices.
//!
//! ## Bias Correction
//!
//! The bias correction terms `(1 - β^t)` are critical. Without them, m and v
//! are biased toward zero during early training steps. The correction ensures
//! the optimizer works well from the first step.
//!
//! ## Implementation Notes
//!
//! This implementation mirrors the GPT-2 model structure exactly:
//! - Separate moment vectors for each parameter
//! - Parallel updates using Rayon for performance
//! - Automatic fallback to sequential for small tensors
//!
//! ## Example
//!
//! ```rust,no_run
//! use crate::optimizer::{AdamWOptimizer, adamw_update};
//! use crate::gpt2_trainable::TrainableGPT2;
//! # use crate::Config;
//!
//! # let config = Config::tiny(512);
//! # let mut model = TrainableGPT2::new(&config);
//! // Initialize optimizer
//! let mut optimizer = AdamWOptimizer::new(&model);
//!
//! // Training loop
//! # let num_steps = 100;
//! for step in 0..num_steps {
//!     // ... forward pass, compute gradients ...
//!     # let grads = todo!();
//!
//!     // Update parameters with weight decay
//!     adamw_update(&mut model, &grads, &mut optimizer, 3e-4, 0.1);
//! }
//! ```
//!
//! ## References
//!
//! - Loshchilov & Hutter (2019): "Decoupled Weight Decay Regularization"
//!   <https://arxiv.org/abs/1711.05101>
//! - Kingma & Ba (2014): "Adam: A Method for Stochastic Optimization"
//!   <https://arxiv.org/abs/1412.6980>
//!
//! # A guided tour (tutorial)
//!
//! This file is written to be read top to bottom.
//!
//! 1. This header: the ideas behind the optimizer (below).
//! 2. The state structs (`AdamWOptimizer`, `BlockAdamState`,
//!    `AttentionAdamState`, `MLPAdamState`): where the "memory" lives.
//! 3. `AdamWOptimizer::new` (and `.clone()` for checkpoints): creating and copying state.
//! 4. `adamw_update` / `adamw_update_with_options`: apply the algorithm to every
//!    parameter. The algorithm itself, for ONE tensor, is the pure function
//!    `adamw_update_tensor` in `optimizer_functions.rs`, which also holds the
//!    unit tests for the arithmetic.
//!
//! ## Where an optimizer fits in training
//!
//! Training a model is a loop of four steps. This file is step 4:
//!
//! ```text
//!   1. FORWARD   run the model on a batch of text    -> predictions, loss
//!   2. BACKWARD  backpropagation                     -> gradients (one per parameter)
//!   3. (clip, schedule the learning rate, ...)
//!   4. OPTIMIZE  adamw_update(model, grads, ...)     -> parameters change a little
//!      then repeat from 1 with the next batch
//! ```
//!
//! A **gradient** `g` tells us, for ONE number in the model, which direction
//! to nudge it to make the loss smaller. The simplest optimizer, plain
//! gradient descent, nudges by `θ = θ - lr * g`. Adam and AdamW are cleverer
//! about HOW MUCH to nudge each number.
//!
//! ## The three ideas inside AdamW, one at a time
//!
//! **1. Momentum (the first moment `m`).** A single batch's gradient is noisy:
//! the loss on one batch of text is only a rough guide. Momentum keeps a
//! running average of recent gradients, like a heavy ball rolling downhill
//! that is not thrown around by every bump.
//!
//! ```text
//! m = β₁ * m + (1 - β₁) * g       β₁ = 0.9: keep 90% of the old average,
//!                                          mix in 10% of the new gradient
//! ```
//!
//! **2. Per-parameter step size (the second moment `v`).** Some parameters
//! see large gradients and others tiny ones. `v` tracks the running average
//! of the SQUARED gradient, a measure of how big this parameter's gradients
//! usually are. Dividing by its square root gives every parameter a step of
//! about the same size, whatever the scale of its gradients:
//!
//! ```text
//! v = β₂ * v + (1 - β₂) * g²      β₂ = 0.95
//! step = lr * m / (√v + ε)        ε = 1e-8 stops a division by zero
//! ```
//!
//! **3. Decoupled weight decay.** Weight decay gently shrinks weights toward
//! zero each step, which discourages the model from relying on a few huge
//! weights. "Decoupled" means the shrinking is a separate multiplication,
//! `θ = θ * (1 - lr * λ)`, NOT mixed into the gradient where `v` would
//! distort it. That separation is the whole difference between Adam (with L2
//! penalty) and AdamW. (Strictly, with an adaptive optimizer decoupled weight
//! decay is no longer the same thing as an L2 penalty; the paper shows the
//! decoupled form generalizes better.)
//!
//! ## Worked example: why bias correction matters
//!
//! `m` and `v` start at 0, so the first averages are pulled toward zero.
//! Take a constant gradient `g = 0.5`, with `β₁ = 0.9`, `β₂ = 0.95`, `lr = 1`:
//!
//! ```text
//! step t = 1
//!   m = 0.9*0 + 0.1*0.5        = 0.05       (the true average is 0.5!)
//!   v = 0.95*0 + 0.05*0.25     = 0.0125     (the true average is 0.25!)
//!
//!   WITHOUT correction: step = m / √v = 0.05 / 0.1118 = 0.447   <- too small
//!
//!   bias_correction1 = 1 - 0.9^1  = 0.1
//!   bias_correction2 = 1 - 0.95^1 = 0.05
//!   m_hat = 0.05  / 0.1  = 0.5
//!   v_hat = 0.0125 / 0.05 = 0.25
//!   WITH correction:    step = m_hat / √v_hat = 0.5 / 0.5 = 1.0     <- right
//! ```
//!
//! With a steady gradient, every step moves the parameter by about `lr`,
//! whatever the gradient's size. With a noisy gradient that flips sign, the
//! momentum averages out and steps shrink (computed with `lr = 1`):
//!
//! ```text
//! gradient:   +1      -1       +1       -1
//! step:      +1.000  -0.053   +0.336   -0.053
//! ```
//!
//! ## What "t" does, and why `step` is incremented first
//!
//! `t` is the number of updates so far. `adamw_update` increments it BEFORE
//! using it, so the first update uses `t = 1`. Using `t = 0` would make
//! `1 - β⁰ = 0` and divide by zero. As `t` grows, `β^t` shrinks toward 0, so
//! the correction factors approach 1 and the correction fades away.
//!
//! ## Weight decay: which parameters get it in THIS file
//!
//! The header says decay applies to "2D weight matrices". In this code it is
//! decided by an explicit `true`/`false` at each call, not by looking at the
//! tensor's shape:
//!
//! | Parameter                                   | Decay? |
//! |---------------------------------------------|--------|
//! | Q, K, V, output projection weights          | yes    |
//! | MLP `fc1`, `fc2` weights                    | yes    |
//! | Final output weight                         | yes    |
//! | Token and position embeddings (they ARE 2D) | no, unless `decay_embeddings = true` |
//! | All biases, all LayerNorm gamma/beta        | no     |
//!
//! Embeddings are excluded by default even though they are 2-D. Call
//! `adamw_update_with_options(.., decay_embeddings = true)` to decay them as
//! well (the original GPT-2 / nanoGPT recipe decays every 2-D tensor).
//!
//! ## Bad gradients
//!
//! A `NaN` or infinite gradient would be remembered forever by the running
//! averages `m` and `v`, so `adamw_update` panics on one, naming the parameter,
//! instead of letting training quietly turn into `NaN`.
//!
//! ## Memory cost
//!
//! AdamW keeps TWO extra numbers (`m` and `v`) for every parameter. Counting
//! the parameters and the gradients too, training holds about four copies of
//! the model. For a 124-million-parameter GPT-2 in `f32` (4 bytes), one copy
//! is about 0.5 GB, so parameters + gradients + `m` + `v` is about 2 GB.
//!
//! ## Glossary
//!
//! - **Parameter / weight:** a number the model learns.
//! - **Gradient:** the loss's slope with respect to one parameter.
//! - **Learning rate (lr, α):** the overall step size.
//! - **Moment:** a running average; the first moment is of `g`, the second of `g²`.
//! - **Epoch / step:** one pass over all data / one call to `adamw_update`.
//! - **Hyperparameter:** a setting chosen by the person (lr, β₁, β₂, λ, ε).

use crate::gpt2_trainable::{GPT2Gradients, TrainableGPT2};
use crate::optimizer_functions::{StepConstants, adamw_update_tensor, zeros_like};
use crate::tensor::Tensor;

/// AdamW optimizer state
///
/// Maintains first and second moment estimates (m and v) for all model parameters.
/// The structure mirrors `GPT2Gradients` to ensure every parameter has corresponding
/// optimizer state.
///
/// # Fields
///
/// - **m_* (first moment)**: Exponential moving average of gradients (momentum)
/// - **v_* (second moment)**: Exponential moving average of squared gradients (variance)
/// - **beta1**: Momentum decay rate (default: 0.9)
/// - **beta2**: Variance decay rate (default: 0.95, lower than Adam's 0.999)
/// - **epsilon**: Numerical stability constant (default: 1e-8)
/// - **step**: Training step count (for bias correction)
///
/// # Why this struct is so large
///
/// The optimizer needs one `m` tensor and one `v` tensor for EVERY parameter
/// tensor in the model, each with the same shape as that parameter. Rust has
/// no way to say "a copy of the model, but with a different meaning", so the
/// state is written out field by field in the same layout as the model and as
/// `GPT2Gradients`. That makes `adamw_update` a simple, aligned walk over
/// (parameter, gradient, m, v) quadruples.
///
/// ```text
/// model parameter    gradient           m (momentum)         v (variance)
/// token_embedding    token_embedding    m_token_embedding    v_token_embedding
/// blocks[0].ln1 ...  block_grads[0] ... m_block_states[0] ... v_block_states[0] ...
/// ```
///
/// # What is NOT stored here
///
/// The learning rate and weight decay are passed to each `adamw_update` call
/// instead of being stored. That lets the caller change the learning rate on
/// every step, which is how learning-rate schedules (warmup, cosine decay)
/// work.
///
/// # Saving and resuming
///
/// When training is paused and resumed, this state must be restored along
/// with the model. Restoring only the model resets `m`, `v` and `step`, which
/// causes a temporary jolt in training.
///
/// # Copying for checkpoints
///
/// `optimizer.clone()` makes a full, independent copy (every tensor is
/// duplicated, including `step`), which is what a checkpoint needs. It is a
/// deep copy, about twice the size of the model's parameters, so avoid doing it
/// every step. Keeping `step` matters: the bias correction depends on it when
/// training resumes.
#[derive(Clone)]
pub struct AdamWOptimizer {
    // First moment (momentum) - matches GPT2Gradients structure
    //
    // Each tensor here has the same shape as the parameter it belongs to.
    pub m_token_embedding: Tensor,
    pub m_position_embedding: Tensor,
    pub m_block_states: Vec<BlockAdamState>,
    pub m_ln_final_gamma: Tensor,
    pub m_ln_final_beta: Tensor,
    pub m_output_weight: Tensor,

    // Second moment (variance) - matches GPT2Gradients structure
    //
    // Same layout again, holding the running average of the squared gradient.
    pub v_token_embedding: Tensor,
    pub v_position_embedding: Tensor,
    pub v_block_states: Vec<BlockAdamState>,
    pub v_ln_final_gamma: Tensor,
    pub v_ln_final_beta: Tensor,
    pub v_output_weight: Tensor,

    // Hyperparameters
    //
    // Fixed settings chosen by the person, not learned.
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    // Number of updates done so far; 0 before the first call. Used for the
    // bias-correction terms.
    pub step: usize,
}

/// Optimizer state for a single transformer block
///
/// Stores moment estimates for all parameters in a transformer block:
/// - Two LayerNorm layers (scale and shift parameters)
/// - Self-attention mechanism (Q, K, V, output projections)
/// - MLP feedforward network (two linear layers)
///
/// The same struct type is used for BOTH the first-moment (`m`) and the
/// second-moment (`v`) copy of a block; `AdamWOptimizer` holds one list of
/// each. A block has this layout (pre-LayerNorm transformer):
///
/// ```text
/// input -> ln1 -> attention -> (+ input) -> ln2 -> mlp -> (+ previous) -> output
/// ```
#[derive(Clone)]
pub struct BlockAdamState {
    // LayerNorm before attention: a scale (gamma) and a shift (beta), 1-D.
    pub ln1_gamma: Tensor,
    pub ln1_beta: Tensor,
    pub attn: AttentionAdamState,
    // LayerNorm before the MLP.
    pub ln2_gamma: Tensor,
    pub ln2_beta: Tensor,
    pub mlp: MLPAdamState,
}

/// Optimizer state for attention mechanism
///
/// Stores moment estimates for all attention parameters:
/// - Q, K, V projection matrices and biases
/// - Output projection matrix and bias
///
/// Attention turns each token into a Query ("what am I looking for?"), a
/// Key ("what do I contain?") and a Value ("what do I pass along?"), each
/// with its own linear layer. The output projection mixes the heads back
/// together. A `*_weight` is a 2-D matrix and a `*_bias` is a 1-D vector.
#[derive(Clone)]
pub struct AttentionAdamState {
    pub q_weight: Tensor,
    pub q_bias: Tensor,
    pub k_weight: Tensor,
    pub k_bias: Tensor,
    pub v_weight: Tensor,
    pub v_bias: Tensor,
    pub out_weight: Tensor,
    pub out_bias: Tensor,
}

/// Optimizer state for MLP (feedforward network)
///
/// Stores moment estimates for the two linear layers:
/// - fc1: First projection (n_embd → 4*n_embd)
/// - fc2: Second projection (4*n_embd → n_embd)
///
/// The MLP widens each token's vector to four times its size, applies a
/// non-linearity, and narrows it back. These two matrices hold most of a
/// transformer's parameters.
#[derive(Clone)]
pub struct MLPAdamState {
    pub fc1_weight: Tensor,
    pub fc1_bias: Tensor,
    pub fc2_weight: Tensor,
    pub fc2_bias: Tensor,
}

impl AdamWOptimizer {
    /// Create a new AdamW optimizer for the given model
    ///
    /// Initializes all moment estimates to zero. The optimizer state mirrors
    /// the model structure exactly, ensuring every parameter has optimizer state.
    ///
    /// # Arguments
    ///
    /// * `model` - The model to optimize
    ///
    /// # Returns
    ///
    /// Optimizer with:
    /// - All moment estimates initialized to zero
    /// - Standard hyperparameters (β₁=0.9, β₂=0.95, ε=1e-8)
    /// - Step counter at 0
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use crate::optimizer::AdamWOptimizer;
    /// # use crate::gpt2_trainable::TrainableGPT2;
    /// # use crate::Config;
    /// let config = Config::tiny(512);
    /// let model = TrainableGPT2::new(&config);
    /// let optimizer = AdamWOptimizer::new(&model);
    /// ```
    ///
    /// # Why zeros?
    ///
    /// Before any training there is no gradient history, so both averages
    /// start empty. The bias-correction terms in `adamw_update` exist
    /// precisely to compensate for this zero start.
    ///
    /// Only each parameter's SHAPE is read from the model, never its values.
    pub fn new(model: &TrainableGPT2) -> Self {
        // Initialize all momentum and variance tensors to zero
        //
        // `zeros_like(&x)` makes a tensor of zeros with the same shape as `x`,
        // reading only the shape (never the values) of the model's parameter.
        // Top-level parameters first: embeddings, final LayerNorm, output weight.
        let m_token_embedding = zeros_like(&model.token_embedding);
        let m_position_embedding = zeros_like(&model.position_embedding);
        let m_ln_final_gamma = zeros_like(&model.ln_final.gamma);
        let m_ln_final_beta = zeros_like(&model.ln_final.beta);
        let m_output_weight = zeros_like(&model.output_weight);

        // The same again for the second moment (v).
        let v_token_embedding = zeros_like(&model.token_embedding);
        let v_position_embedding = zeros_like(&model.position_embedding);
        let v_ln_final_gamma = zeros_like(&model.ln_final.gamma);
        let v_ln_final_beta = zeros_like(&model.ln_final.beta);
        let v_output_weight = zeros_like(&model.output_weight);

        let mut m_block_states = Vec::new();
        let mut v_block_states = Vec::new();

        // One pair of state structs (m and v) for each transformer block, in
        // the same order as `model.blocks`. The order matters: `adamw_update`
        // later pairs block i of the model with state i by position.
        for block in &model.blocks {
            // First-moment state for this block.
            let m_block = BlockAdamState {
                ln1_gamma: zeros_like(&block.ln1.gamma),
                ln1_beta: zeros_like(&block.ln1.beta),
                attn: AttentionAdamState {
                    q_weight: zeros_like(&block.attn.q_proj.weight),
                    q_bias: zeros_like(&block.attn.q_proj.bias),
                    k_weight: zeros_like(&block.attn.k_proj.weight),
                    k_bias: zeros_like(&block.attn.k_proj.bias),
                    v_weight: zeros_like(&block.attn.v_proj.weight),
                    v_bias: zeros_like(&block.attn.v_proj.bias),
                    out_weight: zeros_like(&block.attn.out_proj.weight),
                    out_bias: zeros_like(&block.attn.out_proj.bias),
                },
                ln2_gamma: zeros_like(&block.ln2.gamma),
                ln2_beta: zeros_like(&block.ln2.beta),
                mlp: MLPAdamState {
                    fc1_weight: zeros_like(&block.mlp.fc1.weight),
                    fc1_bias: zeros_like(&block.mlp.fc1.bias),
                    fc2_weight: zeros_like(&block.mlp.fc2.weight),
                    fc2_bias: zeros_like(&block.mlp.fc2.bias),
                },
            };

            // Second-moment state for this block. It starts as an exact copy of
            // the first-moment state (both are all zeros of the same shapes).
            let v_block = m_block.clone();

            m_block_states.push(m_block);
            v_block_states.push(v_block);
        }

        // Assemble the finished optimizer. Field-init shorthand (`m_token_embedding,`)
        // means "the field with this name gets the local variable of the same name".
        Self {
            m_token_embedding,
            m_position_embedding,
            m_block_states,
            m_ln_final_gamma,
            m_ln_final_beta,
            m_output_weight,
            v_token_embedding,
            v_position_embedding,
            v_block_states,
            v_ln_final_gamma,
            v_ln_final_beta,
            v_output_weight,
            beta1: 0.9,
            beta2: 0.95, // Lower than Adam's 0.999, standard for transformers
            epsilon: 1e-8,
            step: 0,
        }
    }
}

/// AdamW optimizer parameter update
///
/// Implements the complete AdamW algorithm with decoupled weight decay:
/// 1. Applies weight decay directly to weight matrices (not biases/LayerNorm)
/// 2. Updates moment estimates (m and v)
/// 3. Applies bias correction
/// 4. Updates model parameters using adaptive learning rates
///
/// # Arguments
///
/// * `model` - Model to update (modified in place)
/// * `grads` - Gradients from backpropagation
/// * `optimizer` - Optimizer state (moment estimates updated in place)
/// * `lr` - Learning rate (alpha in the AdamW paper)
/// * `weight_decay` - Weight decay coefficient (lambda, typically 0.1)
///
/// # Algorithm
///
/// For each parameter θ with gradient g:
/// ```text
/// θ = θ * (1 - lr * weight_decay)  # Weight decay (weight matrices only)
/// m = β₁*m + (1-β₁)*g              # Update momentum
/// v = β₂*v + (1-β₂)*g²             # Update variance
/// m_hat = m / (1 - β₁^step)        # Bias correction
/// v_hat = v / (1 - β₂^step)        # Bias correction
/// θ -= lr * m_hat / (√v_hat + ε)   # Update parameter
/// ```
///
/// # Selective Weight Decay
///
/// Weight decay is applied to the weight matrices of the linear layers
/// (attention Q/K/V/output, MLP fc1/fc2, final output weight), not to:
/// - 1D tensors (biases, LayerNorm parameters)
/// - Embeddings (following common practice; use
///   [`adamw_update_with_options`] to decay them too)
///
/// # Performance
///
/// Uses parallel computation via Rayon for tensors with >1000 elements.
/// Small tensors use sequential updates to avoid parallelization overhead.
///
/// # Example
///
/// ```rust,no_run
/// # use crate::optimizer::{AdamWOptimizer, adamw_update};
/// # use crate::gpt2_trainable::TrainableGPT2;
/// # use crate::Config;
/// # let config = Config::tiny(512);
/// # let mut model = TrainableGPT2::new(&config);
/// # let grads = todo!();
/// let mut optimizer = AdamWOptimizer::new(&model);
///
/// // Training step with weight decay
/// adamw_update(&mut model, &grads, &mut optimizer, 3e-4, 0.1);
/// ```
///
/// # Numbers to try by hand
///
/// One weight with `θ = 1.0`, `g = 0.5`, `lr = 3e-4`, `λ = 0.1`, first step:
///
/// ```text
/// decay factor = 1 - lr*λ = 1 - 0.00003 = 0.99997      θ becomes 0.99997
/// m = 0.1 * 0.5            = 0.05      m_hat = 0.05 / 0.1  = 0.5
/// v = 0.05 * 0.25          = 0.0125    v_hat = 0.0125/0.05 = 0.25
/// step = lr * 0.5 / (0.5 + 1e-8)       = 3e-4  (about lr)
/// θ = 0.99997 - 0.0003                 = 0.99967
/// ```
///
/// The decay (3e-5) is a tiny part of the move; the Adam step dominates.
/// The decay matters through accumulation over many thousands of steps.
/// (The unit tests at the bottom of this file check these exact numbers.)
///
/// # Panics
///
/// Panics, naming the parameter, if any gradient or optimizer-state tensor has a
/// different shape from its parameter (see `adamw_update_tensor`). The model,
/// gradients and optimizer must come from the same configuration.
///
/// Also panics, naming the parameter, if a gradient contains `NaN` or infinity.
/// Without this check, one bad value would poison that parameter's `m` and `v`
/// for the rest of training, because the averages remember it. The usual causes
/// are a learning rate that is too high or a loss that overflowed. The panic
/// leaves earlier parameters of the same step already updated, so reload the
/// last checkpoint rather than continuing with that model.
///
/// # Order of the parameters
///
/// The parameters are updated in this order: embeddings, then each block (in
/// order), then the final LayerNorm, then the output weight. The order does
/// not affect the result, because every parameter is updated independently.
pub fn adamw_update(
    model: &mut TrainableGPT2,
    grads: &GPT2Gradients,
    optimizer: &mut AdamWOptimizer,
    lr: f32,
    weight_decay: f32,
) {
    // The standard behavior: embeddings are NOT weight-decayed.
    adamw_update_with_options(model, grads, optimizer, lr, weight_decay, false);
}

/// AdamW update with a choice about decaying the embeddings
///
/// Identical to [`adamw_update`], except that `decay_embeddings` controls
/// whether the token and position embeddings are weight-decayed too:
///
/// - `false`: embeddings are exempt (what `adamw_update` does).
/// - `true`: embeddings are decayed like the other weight matrices. The original
///   GPT-2 / nanoGPT recipe decays every 2-D tensor, including embeddings.
///
/// Biases and LayerNorm parameters are never decayed.
///
/// # Arguments
///
/// * `decay_embeddings` - Whether to apply weight decay to the embeddings
///
/// See [`adamw_update`] for the other arguments, the algorithm and the panics.
pub fn adamw_update_with_options(
    model: &mut TrainableGPT2,
    grads: &GPT2Gradients,
    optimizer: &mut AdamWOptimizer,
    lr: f32,
    weight_decay: f32,
    decay_embeddings: bool,
) {
    // Count this update BEFORE using it, so the first update has t = 1.
    // (With t = 0 the bias correction `1 - β^0` would be 0: division by zero.)
    optimizer.step += 1;

    // Everything that is the same for all parameters this step, including the
    // bias-correction factors. Copying the settings out of `optimizer` here also
    // avoids reading `optimizer` while its `m_*`/`v_*` tensors are borrowed
    // mutably below.
    let c = StepConstants::new(
        lr,
        weight_decay,
        optimizer.beta1,
        optimizer.beta2,
        optimizer.epsilon,
        optimizer.step,
    );

    // Update embeddings (no weight decay by default - common practice)
    //
    // Token embedding: one row of numbers per vocabulary entry (a lookup table
    // from token ID to vector). Position embedding: one row per position in
    // the sentence. Both are 2-D; whether they decay is the caller's choice.
    adamw_update_tensor(
        "token_embedding",
        &mut model.token_embedding,
        &grads.token_embedding,
        &mut optimizer.m_token_embedding,
        &mut optimizer.v_token_embedding,
        &c,
        decay_embeddings,
    );
    adamw_update_tensor(
        "position_embedding",
        &mut model.position_embedding,
        &grads.position_embedding,
        &mut optimizer.m_position_embedding,
        &mut optimizer.v_position_embedding,
        &c,
        decay_embeddings,
    );

    // The block lists must be the same length, or `zip` below would silently
    // skip blocks.
    assert!(
        model.blocks.len() == grads.block_grads.len()
            && model.blocks.len() == optimizer.m_block_states.len()
            && model.blocks.len() == optimizer.v_block_states.len(),
        "adamw_update: model has {} blocks but gradients have {} and optimizer state has {}/{}",
        model.blocks.len(),
        grads.block_grads.len(),
        optimizer.m_block_states.len(),
        optimizer.v_block_states.len()
    );

    // Update all transformer blocks
    //
    // Walk three lists in lock-step: the model's blocks (mutably), the
    // gradients for each block, and the (m, v) state for each block. `zip`
    // pairs them by position, which is why `new` built them in model order.
    // The borrow checker accepts this because `m_block_states` and
    // `v_block_states` are different fields: each is borrowed mutably once.
    for ((block, block_grads), (m_block, v_block)) in
        model.blocks.iter_mut().zip(&grads.block_grads).zip(
            optimizer
                .m_block_states
                .iter_mut()
                .zip(optimizer.v_block_states.iter_mut()),
        )
    {
        // LayerNorm 1 (no decay on 1D scale/shift parameters)
        //
        // LayerNorm rescales a token's vector to a steady size, then applies a
        // learned scale (gamma) and shift (beta). Decaying gamma toward 0 would
        // fight the layer's purpose, so these are exempt.
        adamw_update_tensor(
            "block.ln1.gamma",
            &mut block.ln1.gamma,
            &block_grads.ln1_gamma,
            &mut m_block.ln1_gamma,
            &mut v_block.ln1_gamma,
            &c,
            false, // No decay on LayerNorm
        );
        adamw_update_tensor(
            "block.ln1.beta",
            &mut block.ln1.beta,
            &block_grads.ln1_beta,
            &mut m_block.ln1_beta,
            &mut v_block.ln1_beta,
            &c,
            false, // No decay on LayerNorm
        );

        // Self-attention (decay on 2D weight matrices, not on 1D biases)
        //
        // Query, key, value and output projections: for each, the weight
        // matrix is decayed (`true`) and the bias vector is not (`false`).
        adamw_update_tensor(
            "block.attn.q_proj.weight",
            &mut block.attn.q_proj.weight,
            &block_grads.attn.q_weight,
            &mut m_block.attn.q_weight,
            &mut v_block.attn.q_weight,
            &c,
            true, // Decay on weight matrix
        );
        adamw_update_tensor(
            "block.attn.q_proj.bias",
            &mut block.attn.q_proj.bias,
            &block_grads.attn.q_bias,
            &mut m_block.attn.q_bias,
            &mut v_block.attn.q_bias,
            &c,
            false, // No decay on bias
        );
        adamw_update_tensor(
            "block.attn.k_proj.weight",
            &mut block.attn.k_proj.weight,
            &block_grads.attn.k_weight,
            &mut m_block.attn.k_weight,
            &mut v_block.attn.k_weight,
            &c,
            true, // Decay on weight matrix
        );
        adamw_update_tensor(
            "block.attn.k_proj.bias",
            &mut block.attn.k_proj.bias,
            &block_grads.attn.k_bias,
            &mut m_block.attn.k_bias,
            &mut v_block.attn.k_bias,
            &c,
            false, // No decay on bias
        );
        adamw_update_tensor(
            "block.attn.v_proj.weight",
            &mut block.attn.v_proj.weight,
            &block_grads.attn.v_weight,
            &mut m_block.attn.v_weight,
            &mut v_block.attn.v_weight,
            &c,
            true, // Decay on weight matrix
        );
        adamw_update_tensor(
            "block.attn.v_proj.bias",
            &mut block.attn.v_proj.bias,
            &block_grads.attn.v_bias,
            &mut m_block.attn.v_bias,
            &mut v_block.attn.v_bias,
            &c,
            false, // No decay on bias
        );
        adamw_update_tensor(
            "block.attn.out_proj.weight",
            &mut block.attn.out_proj.weight,
            &block_grads.attn.out_weight,
            &mut m_block.attn.out_weight,
            &mut v_block.attn.out_weight,
            &c,
            true, // Decay on weight matrix
        );
        adamw_update_tensor(
            "block.attn.out_proj.bias",
            &mut block.attn.out_proj.bias,
            &block_grads.attn.out_bias,
            &mut m_block.attn.out_bias,
            &mut v_block.attn.out_bias,
            &c,
            false, // No decay on bias
        );

        // LayerNorm 2 (no decay on 1D scale/shift parameters)
        adamw_update_tensor(
            "block.ln2.gamma",
            &mut block.ln2.gamma,
            &block_grads.ln2_gamma,
            &mut m_block.ln2_gamma,
            &mut v_block.ln2_gamma,
            &c,
            false, // No decay on LayerNorm
        );
        adamw_update_tensor(
            "block.ln2.beta",
            &mut block.ln2.beta,
            &block_grads.ln2_beta,
            &mut m_block.ln2_beta,
            &mut v_block.ln2_beta,
            &c,
            false, // No decay on LayerNorm
        );

        // MLP (decay on 2D weight matrices, not on 1D biases)
        //
        // fc1 widens (n_embd -> 4*n_embd) and fc2 narrows it back. These are
        // the largest matrices in a block, so they are the ones that take the
        // parallel path.
        adamw_update_tensor(
            "block.mlp.fc1.weight",
            &mut block.mlp.fc1.weight,
            &block_grads.mlp.fc1_weight,
            &mut m_block.mlp.fc1_weight,
            &mut v_block.mlp.fc1_weight,
            &c,
            true, // Decay on weight matrix
        );
        adamw_update_tensor(
            "block.mlp.fc1.bias",
            &mut block.mlp.fc1.bias,
            &block_grads.mlp.fc1_bias,
            &mut m_block.mlp.fc1_bias,
            &mut v_block.mlp.fc1_bias,
            &c,
            false, // No decay on bias
        );
        adamw_update_tensor(
            "block.mlp.fc2.weight",
            &mut block.mlp.fc2.weight,
            &block_grads.mlp.fc2_weight,
            &mut m_block.mlp.fc2_weight,
            &mut v_block.mlp.fc2_weight,
            &c,
            true, // Decay on weight matrix
        );
        adamw_update_tensor(
            "block.mlp.fc2.bias",
            &mut block.mlp.fc2.bias,
            &block_grads.mlp.fc2_bias,
            &mut m_block.mlp.fc2_bias,
            &mut v_block.mlp.fc2_bias,
            &c,
            false, // No decay on bias
        );
    }

    // Final layer norm (no decay on 1D scale/shift parameters)
    //
    // The last LayerNorm, applied after all the blocks and before the output
    // layer.
    adamw_update_tensor(
        "ln_final.gamma",
        &mut model.ln_final.gamma,
        &grads.ln_final_gamma,
        &mut optimizer.m_ln_final_gamma,
        &mut optimizer.v_ln_final_gamma,
        &c,
        false, // No decay on LayerNorm
    );
    adamw_update_tensor(
        "ln_final.beta",
        &mut model.ln_final.beta,
        &grads.ln_final_beta,
        &mut optimizer.m_ln_final_beta,
        &mut optimizer.v_ln_final_beta,
        &c,
        false, // No decay on LayerNorm
    );

    // Output projection weight (decay on 2D weight matrix)
    //
    // Maps the final vector of each token to one score per vocabulary entry
    // (the "logits" that softmax turns into next-token probabilities).
    adamw_update_tensor(
        "output_weight",
        &mut model.output_weight,
        &grads.output_weight,
        &mut optimizer.m_output_weight,
        &mut optimizer.v_output_weight,
        &c,
        true, // Decay on weight matrix
    );
}
