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

# Default loss configuration
const DEFAULT_LOSS_CONFIG = (loss=Flux.mse, agg=StatsBase.mean)

"""
    create_masked_loss_function(loss_config)

Create a masked loss function from a configuration named tuple.

# Arguments
- `loss_config`: Named tuple with `loss` and `agg` fields: `(loss=loss_function, agg=aggregation_function)`

# Returns
- Function that computes masked loss: `f(predictions, targets, mask)`

# Examples
```julia
# Create MSE loss with mean aggregation
loss_fn = create_masked_loss_function((loss=Flux.mse, agg=StatsBase.mean))

# Create MAE loss with sum aggregation
loss_fn = create_masked_loss_function((loss=Flux.mae, agg=sum))

# Use in training
loss = loss_fn(predictions, targets, mask)
```
"""
function create_masked_loss_function(loss_config::NamedTuple{(:loss, :agg), <:Tuple{<:Function, <:Function}})
    return (predictions, targets, mask) -> masked_loss(predictions, targets, mask, loss_config.loss, loss_config.agg)
end

# Convenience function for backward compatibility
function masked_mse(predictions, targets, mask)
    return masked_loss(predictions, targets, mask, Flux.mse, StatsBase.mean)
end
