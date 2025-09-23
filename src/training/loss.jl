
"""
    masked_mse(predictions, targets, mask)

Compute mean squared error only on valid (masked) entries.

# Arguments
- `predictions`: Model predictions
- `targets`: Ground truth targets  
- `mask`: Boolean mask indicating valid entries

# Returns
- MSE computed only on valid entries specified by mask
"""
function masked_mse(predictions, targets, mask)
    # Only compute loss on valid (non-NaN) entries
    valid_predictions = @view predictions[mask]
    valid_targets = @view targets[mask]
    
    # Return mean squared error only for valid entries
    return Flux.mse(valid_predictions, valid_targets; agg = mean)
end
