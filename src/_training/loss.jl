"""
    masked_loss(predictions, targets, mask, loss_fcn, agg)

Apply any Flux loss function only on valid (masked) entries.

# Arguments
- `predictions`: Model predictions
- `targets`: Ground truth targets  
- `mask`: Boolean mask indicating valid entries
- `loss_fcn`: Flux loss function (e.g., Flux.mse, Flux.mae, Flux.huber_loss)
- `agg`: Aggregation function (default: StatsBase.mean)

# Returns
- Loss computed only on valid entries specified by mask

# Examples
```julia
# MSE with mean aggregation (default)
loss = masked_loss(ŷ, y, mask, Flux.mse, StatsBase.mean)

# MAE with sum aggregation  
loss = masked_loss(ŷ, y, mask, Flux.mae, sum)

# Huber loss with mean aggregation
loss = masked_loss(ŷ, y, mask, Flux.huber_loss, StatsBase.mean)
```
"""
function masked_loss(predictions, targets, mask, loss_fcn, agg=StatsBase.mean)
    # Only compute loss on valid (non-NaN) entries
    valid_predictions = @view predictions[mask]
    valid_targets = @view targets[mask]
    
    # Apply the loss function with specified aggregation
    return loss_fcn(valid_predictions, valid_targets; agg=agg)
end

# Default loss specification (user-facing config)
const DEFAULT_LOSS_SPEC = (loss=Flux.mse, agg=StatsBase.mean)

"""
    compile_loss(loss_spec) -> compiled_loss(predictions, targets, mask)

Compile a loss specification (NamedTuple) into a callable 3-arg loss function.

# Arguments
- `loss_spec`: Named tuple `(loss=loss_function, agg=aggregation_function)`

# Returns
- `compiled_loss`: Function `(predictions, targets, mask) -> scalar`

# Examples
```julia
# Compile MSE loss with mean aggregation
compiled = compile_loss((loss=Flux.mse, agg=StatsBase.mean))

# Compile MAE loss with sum aggregation
compiled = compile_loss((loss=Flux.mae, agg=sum))

# Use in training
loss = compiled(predictions, targets, mask)
```
"""
function compile_loss(loss_spec::NamedTuple{(:loss, :agg), <:Tuple{<:Function, <:Function}})
    return (predictions, targets, mask) -> masked_loss(predictions, targets, mask, loss_spec.loss, loss_spec.agg)
end

# Backward compatibility alias
const create_masked_loss_function = compile_loss

# Convenience function for backward compatibility
function masked_mse(predictions, targets, mask)
    return masked_loss(predictions, targets, mask, Flux.mse, StatsBase.mean)
end
