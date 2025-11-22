# Examples of Custom Loss Functions and Training Extensibility

using Flux, CUDA
using AutoComputationalGraphTuning

## Example 1: Standard Usage (Backward Compatible)
# This still works exactly as before
model, stats, train_stats = train_final_model(
    data, create_model;
    seed=42,
    max_epochs=50
)

## Example 2: Model with Gradient Penalty Regularization
"""
Custom model that can return activations for gradient analysis
"""
struct ModelWithActivations
    backbone
    head
end

Flux.@functor ModelWithActivations

function (m::ModelWithActivations)(x; return_activations=false)
    acts = m.backbone(x)
    preds = m.head(acts)
    return_activations ? (preds, acts) : preds
end

"""
Custom loss with gradient penalty on activations
"""
function gradient_penalty_loss(model, seq, labels, nan_mask; λ_gp=0.1)
    # Forward pass with activations
    result = model(seq; return_activations=true)
    preds, activations = result
    
    # Standard prediction loss
    pred_loss = masked_mse(preds, labels, nan_mask)
    
    # Compute gradient penalty: ∇_activations(preds)
    grad_penalty = sum(abs2, Flux.gradient(a -> sum(model.head(a)), activations)[1])
    
    # Combined loss
    total_loss = pred_loss + λ_gp * grad_penalty
    
    # Return auxiliary info for logging
    aux_info = Dict(
        :pred_loss => pred_loss,
        :grad_penalty => grad_penalty,
        :valid_count => sum(nan_mask),
        :λ_gp => λ_gp
    )
    
    (total_loss, aux_info)
end

# Use it in training
model, stats = train_model(
    model, opt_state, train_dl, val_dl, output_dim;
    compute_loss = (m, x, y, mask) -> gradient_penalty_loss(m, x, y, mask; λ_gp=0.05),
    max_epochs = 50
)

## Example 3: Multi-Task Learning with Weighted Losses
"""
Model that predicts multiple related outputs
"""
function multi_task_loss(model, seq, labels, nan_mask; task_weights=[1.0, 0.5, 0.3], task_ranges=nothing)
    # Model returns predictions for multiple tasks
    all_preds = model(seq)  # Returns tuple or named tuple
    
    # If task_ranges not provided, split labels evenly
    if isnothing(task_ranges)
        n_tasks = length(all_preds)
        dim = size(labels, 1)
        step = dim ÷ n_tasks
        task_ranges = [((i-1)*step+1):(i*step) for i in 1:n_tasks]
    end
    
    # Separate losses for each task
    task_losses = []
    for (i, (preds, weight)) in enumerate(zip(all_preds, task_weights))
        # Each task may have different labels (slice labels appropriately)
        task_labels = labels[task_ranges[i], :]
        task_mask = nan_mask[task_ranges[i], :]
        task_loss = masked_mse(preds, task_labels, task_mask)
        push!(task_losses, weight * task_loss)
    end
    
    total_loss = sum(task_losses)
    
    aux_info = Dict(
        :task_losses => task_losses,
        :total_loss => total_loss,
        :valid_count => sum(nan_mask)
    )
    
    (total_loss, aux_info)
end

## Example 4: Attention Regularization
"""
Encourage sparse attention patterns
"""
function attention_regularized_loss(model, seq, labels, nan_mask; λ_attn=0.01)
    # Model returns (predictions, attention_weights)
    preds, attn_weights = model(seq; return_attention=true)
    
    # Standard loss
    pred_loss = masked_mse(preds, labels, nan_mask)
    
    # Attention sparsity regularization (entropy penalty)
    attn_entropy = -sum(attn_weights .* log.(attn_weights .+ 1e-10))
    
    # Or use L1 regularization for sparsity
    attn_sparsity = sum(abs, attn_weights)
    
    total_loss = pred_loss + λ_attn * attn_sparsity
    
    aux_info = Dict(
        :pred_loss => pred_loss,
        :attn_sparsity => attn_sparsity,
        :attn_entropy => attn_entropy,
        :valid_count => sum(nan_mask)
    )
    
    (total_loss, aux_info)
end

## Example 5: Contrastive Learning Component
"""
Add contrastive loss for representation learning
"""
function contrastive_loss(model, seq, labels, nan_mask; temperature=0.07, λ_contrast=0.1)
    # Forward pass
    preds = model(seq)
    
    # Standard supervised loss
    supervised_loss = masked_mse(preds, labels, nan_mask)
    
    # Get representations (assuming model has .encoder)
    representations = model.encoder(seq)
    
    # Simple contrastive loss (NT-Xent style)
    # Normalize representations
    reps_norm = representations ./ (sqrt.(sum(abs2, representations, dims=1)) .+ 1e-8)
    
    # Compute similarity matrix
    sim_matrix = reps_norm' * reps_norm
    
    # Contrastive loss (maximize similarity of similar samples)
    contrast_loss = -sum(log.(exp.(sim_matrix ./ temperature) ./ sum(exp.(sim_matrix ./ temperature), dims=2)))
    
    total_loss = supervised_loss + λ_contrast * contrast_loss
    
    aux_info = Dict(
        :supervised_loss => supervised_loss,
        :contrast_loss => contrast_loss,
        :valid_count => sum(nan_mask)
    )
    
    (total_loss, aux_info)
end

## Example 6: Gradient Clipping and Monitoring
"""
Monitor gradient norms and clip if necessary
"""
function loss_with_grad_monitoring(model, seq, labels, nan_mask; max_grad_norm=1.0)
    preds = model(seq)
    loss = masked_mse(preds, labels, nan_mask)
    
    # Compute gradients
    grads = Flux.gradient(m -> masked_mse(m(seq), labels, nan_mask), model)[1]
    
    # Compute gradient norm
    grad_norm = sqrt(sum(sum(abs2, g) for g in Flux.params(model)))
    
    # Optional: clip gradients
    if grad_norm > max_grad_norm
        scale = max_grad_norm / grad_norm
        # Apply scaling (this would need to be done in the update step)
    end
    
    aux_info = Dict(
        :loss => loss,
        :grad_norm => grad_norm,
        :clipped => grad_norm > max_grad_norm,
        :valid_count => sum(nan_mask)
    )
    
    (loss, aux_info)
end

## Example 7: Custom Evaluation Metrics During Training
"""
Track additional metrics during training
"""
function loss_with_custom_metrics(model, seq, labels, nan_mask)
    preds = model(seq)
    loss = masked_mse(preds, labels, nan_mask)
    
    # Compute additional metrics
    valid_preds = preds[nan_mask]
    valid_labels = labels[nan_mask]
    
    mae = sum(abs, valid_preds - valid_labels) / length(valid_preds)
    max_error = maximum(abs, valid_preds - valid_labels)
    
    aux_info = Dict(
        :loss => loss,
        :mae => mae,
        :max_error => max_error,
        :valid_count => sum(nan_mask),
        :pred_mean => mean(valid_preds),
        :label_mean => mean(valid_labels)
    )
    
    (loss, aux_info)
end

## Example 8: Variational Autoencoder (VAE) Style Loss
"""
KL divergence regularization
"""
function vae_loss(model, seq, labels, nan_mask; β=0.001)
    # Model returns (predictions, μ, logσ²)
    preds, μ, logσ² = model(seq; return_latent_params=true)
    
    # Reconstruction loss
    recon_loss = masked_mse(preds, labels, nan_mask)
    
    # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0,I)
    kl_loss = -0.5 * sum(1 .+ logσ² .- μ.^2 .- exp.(logσ²))
    
    total_loss = recon_loss + β * kl_loss
    
    aux_info = Dict(
        :recon_loss => recon_loss,
        :kl_loss => kl_loss,
        :β => β,
        :valid_count => sum(nan_mask)
    )
    
    (total_loss, aux_info)
end

## Example 9: Curriculum Learning
"""
Progressively increase task difficulty
"""
mutable struct CurriculumLoss
    epoch::Int
    max_epochs::Int
    easy_weight::Float32
    hard_weight::Float32
end

function (cl::CurriculumLoss)(model, seq, labels, nan_mask)
    preds = model(seq)
    
    # Easy samples: well-defined labels
    easy_mask = nan_mask .& (abs.(labels) .< 1.0)
    easy_loss = masked_mse(preds, labels, easy_mask)
    
    # Hard samples: edge cases
    hard_mask = nan_mask .& (abs.(labels) .>= 1.0)
    hard_loss = masked_mse(preds, labels, hard_mask)
    
    # Gradually shift weight from easy to hard
    α = cl.epoch / cl.max_epochs
    total_loss = (1-α) * easy_loss + α * hard_loss
    
    aux_info = Dict(
        :easy_loss => easy_loss,
        :hard_loss => hard_loss,
        :curriculum_α => α,
        :valid_count => sum(nan_mask)
    )
    
    cl.epoch += 1  # Increment for next batch
    
    (total_loss, aux_info)
end

# Usage
curriculum = CurriculumLoss(1, 100, 1.0, 0.0)
train_model(model, opt_state, train_dl, val_dl, ydim; compute_loss=curriculum)

## Example 10: Combining Multiple Custom Losses
"""
Flexible composition of multiple loss components
"""
struct CompositeLoss
    components::Vector{Tuple{Function, Float32}}  # (loss_fn, weight)
end

function (cl::CompositeLoss)(model, seq, labels, nan_mask)
    total_loss = 0.0f0
    aux_info = Dict{Symbol, Any}()
    
    for (i, (loss_fn, weight)) in enumerate(cl.components)
        component_loss, component_aux = loss_fn(model, seq, labels, nan_mask)
        total_loss += weight * component_loss
        
        # Store component losses
        aux_info[Symbol("component_$(i)_loss")] = component_loss
        aux_info[Symbol("component_$(i)_weight")] = weight
        
        # Merge aux info
        for (k, v) in component_aux
            aux_info[Symbol("component_$(i)_$k")] = v
        end
    end
    
    aux_info[:total_loss] = total_loss
    aux_info[:valid_count] = sum(nan_mask)
    
    (total_loss, aux_info)
end

# Usage: combine MSE + gradient penalty + attention regularization
composite = CompositeLoss([
    ((m,x,y,mask) -> (masked_mse(m(x), y, mask), Dict()), 1.0),
    (gradient_penalty_loss, 0.1),
    (attention_regularized_loss, 0.05)
])

train_model(model, opt_state, train_dl, val_dl, ydim; compute_loss=composite)
