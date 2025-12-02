
"""
    evaluate_validation_loss(model, hp, dataloader)

Evaluate validation loss using masked MSE for NaN handling.

# Arguments
- `model`: Trained model instance
- `hp`: HyperParameters 
- `dataloader`: Validation data loader

# Returns
- Average validation loss over all batches
"""
function evaluate_validation_loss(model, dataloader)
    total_loss = 0.0
    total_batches = 0
    
    for (seq, affs) in dataloader
        seq, affs = seq |> cu, affs |> cu
        nan_mask = .!isnan.(affs)
        
        # Forward pass only (no gradients)
        yhat = model(seq)
        loss = masked_mse(yhat, affs, nan_mask)
        
        total_loss += loss
        total_batches += 1
    end
    
    return total_loss / total_batches
end

"""
    compute_r2_scores(predictions, targets, mask)

Compute R² scores for individual RBPs and aggregated across all predictions.

# Arguments
- `predictions`: Model predictions matrix (n_rbps, n_samples)
- `targets`: Ground truth targets matrix (n_rbps, n_samples)  
- `mask`: Boolean mask for valid entries (n_rbps, n_samples)

# Returns
- `individual_r2`: R² score for each RBP (may contain NaN for insufficient data)
- `aggregated_r2`: Overall R² across all valid predictions
"""
function compute_r2_scores(predictions, targets, mask)
    n_rbps = size(predictions, 1)  # Number of RBPs (rows)
    n_samples = size(predictions, 2)  # Number of samples (columns)
    
    # Individual R² for each RBP (row of affinity matrix)
    individual_r2 = DEFAULT_FLOAT_TYPE[]
    
    for rbp_idx in 1:n_rbps
        # Get mask for this specific RBP across all samples
        rbp_mask = mask[rbp_idx, :]
        
        if sum(rbp_mask) < 2  # Need at least 2 points for R²
            push!(individual_r2, DEFAULT_FLOAT_TYPE(NaN))
            continue
        end
        
        pred_vals = predictions[rbp_idx, rbp_mask]
        true_vals = targets[rbp_idx, rbp_mask]
        
        # R² = 1 - (SS_res / SS_tot)
        ss_res = sum((true_vals .- pred_vals).^2)
        ss_tot = sum((true_vals .- mean(true_vals)).^2)
        
        if ss_tot ≈ 0  # Handle case where all true values are the same
            push!(individual_r2, DEFAULT_FLOAT_TYPE(NaN))
        else
            r2 = 1 - (ss_res / ss_tot)
            push!(individual_r2, r2)
        end
    end
    
    # Aggregated R² across all valid predictions
    valid_mask = mask
    valid_predictions = predictions[valid_mask]
    valid_targets = targets[valid_mask]
    
    if length(valid_predictions) == 0
        aggregated_r2 = DEFAULT_FLOAT_TYPE(NaN)
    else
        ss_res_total = sum((valid_targets .- valid_predictions).^2)
        ss_tot_total = sum((valid_targets .- mean(valid_targets)).^2)
        aggregated_r2 = 1 - (ss_res_total / ss_tot_total)
    end
    
    return individual_r2, aggregated_r2
end

"""
    evaluate_validation_metrics(model, dataloader, n_outputs)

Evaluate comprehensive validation metrics including loss and R² scores.

# Arguments
- `model`: Trained model instance
- `dataloader`: Validation data loader  
- `n_outputs`: Number of output targets (RBPs)

# Returns
- `avg_loss`: Average validation loss
- `individual_r2`: R² score for each RBP
- `aggregated_r2`: Overall R² across all predictions
"""
function evaluate_validation_metrics(model, dataloader, n_outputs)
    total_loss = 0.0
    total_batches = 0
    all_predictions = DEFAULT_FLOAT_TYPE[]
    all_targets = DEFAULT_FLOAT_TYPE[]
    all_masks = Bool[]
    
    for (seq, affs) in dataloader
        seq, affs = seq |> cu, affs |> cu
        nan_mask = .!isnan.(affs)
        
        # Forward pass only (no gradients)
        yhat = model(seq)
        loss = masked_mse(yhat, affs, nan_mask)
        
        total_loss += loss
        total_batches += 1
        
        # Collect predictions for R² calculation
        append!(all_predictions, vec(yhat |> cpu))
        append!(all_targets, vec(affs |> cpu))
        append!(all_masks, vec(nan_mask |> cpu))
    end
    
    avg_loss = total_loss / total_batches
    
    # Reshape for R² calculation
    n_samples = length(all_predictions) ÷ n_outputs
    
    pred_matrix = reshape(all_predictions, n_outputs, n_samples)
    target_matrix = reshape(all_targets, n_outputs, n_samples)
    mask_matrix = reshape(all_masks, n_outputs, n_samples)
    
    individual_r2, aggregated_r2 = compute_r2_scores(pred_matrix, target_matrix, mask_matrix)
    
    return avg_loss, individual_r2, aggregated_r2
end