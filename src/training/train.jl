"""
Train single batch with flexible loss computation.

# Arguments
- `model`: The model to train
- `opt_state`: Optimizer state
- `seq, labels`: Batch data
- `compute_loss`: Function(model, seq, labels, nan_mask) -> (loss, aux_info)
  - Should return loss scalar and optionally auxiliary info dict
  - Default: standard masked MSE loss

# Returns
- `loss`: Scalar loss value
- `aux_info`: Dict with auxiliary information (e.g., valid_count, regularizers, etc.)
"""
function train_batch!(model, opt_state, seq, labels; compute_loss=nothing)
    seq, labels = seq |> cu, labels |> cu
    nan_mask = .!isnan.(labels)
    
    # Default loss computation if none provided
    if isnothing(compute_loss)
        compute_loss = (m, x, y, mask) -> begin
            preds = m(x)
            loss = masked_mse(preds, y, mask)
            (loss, Dict(:valid_count => sum(mask)))
        end
    end
    
    # Compute loss and gradients
    (loss, aux_info), gs = Flux.withgradient(model) do m
        compute_loss(m, seq, labels, nan_mask)
    end
    
    # Update parameters
    Flux.update!(opt_state, model, gs[1])
    
    loss, aux_info
end

"""Train single epoch"""
function train_epoch!(model, opt_state, dataloader, epoch, print_every; compute_loss=nothing)
    epoch_losses = DEFAULT_FLOAT_TYPE[]
    epoch_aux = []

    for (batch_idx, (seq, labels)) in enumerate(dataloader)
        loss, aux = train_batch!(model, opt_state, seq, labels; compute_loss)
        
        push!(epoch_losses, loss)
        push!(epoch_aux, aux)
        
        if batch_idx % print_every == 0
            avg_loss = StatsBase.mean(epoch_losses)
            valid_info = haskey(aux, :valid_count) ? ", Valid: $(aux[:valid_count])" : ""
            println("Epoch $epoch, Batch $batch_idx: Loss = $(round(loss, digits=6)), " * 
                   "Avg = $(round(avg_loss, digits=6))$valid_info")
        end
    end
    
    StatsBase.mean(epoch_losses), epoch_aux
end

# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

"""
Train a model with early stopping and return the best model state and training stats.

# Arguments
- `compute_loss`: Optional custom loss function(model, seq, labels, nan_mask) -> (loss, aux_info)
  - If not provided, uses standard masked MSE
  - Can return auxiliary info dict for logging custom metrics
  - Examples: gradient penalties, attention regularization, multi-task losses

# Returns
- `best_model_state`: State dict of the best model (lowest validation loss)
- `training_stats`: Dict with training history and final metrics

# Example Custom Loss
```julia
# Gradient penalty example
function my_loss(model, seq, labels, mask)
    # Forward pass (model can return multiple outputs)
    output = model(seq)
    preds = output isa Tuple ? output[1] : output
    
    # Standard prediction loss
    pred_loss = masked_mse(preds, labels, mask)
    
    # Custom regularizer (e.g., gradient penalty)
    grad_penalty = compute_gradient_penalty(model, seq)
    
    total_loss = pred_loss + 0.1 * grad_penalty
    aux = Dict(:pred_loss => pred_loss, :grad_penalty => grad_penalty)
    
    (total_loss, aux)
end

# Use it
train_model(model, opt, train_dl, val_dl, ydim; compute_loss=my_loss)
```
"""
function train_model(model, opt_state, train_dl, val_dl, output_dim;
                     max_epochs=50, patience=10, min_delta=1e-4, print_every=100,
                     test_set=false, compute_loss=nothing, loss_fcn=masked_mse)
    
    # Backward compatibility: convert old loss_fcn to new compute_loss
    if isnothing(compute_loss) && !isnothing(loss_fcn)
        compute_loss = (m, x, y, mask) -> begin
            preds = m(x)
            preds = preds isa Tuple ? preds[1] : preds  # Handle tuple outputs
            (loss_fcn(preds, y, mask), Dict(:valid_count => sum(mask)))
        end
    end
    
    # Early stopping variables
    best_val_loss = Inf
    best_r2 = -Inf
    epochs_without_improvement = 0
    best_model_state = nothing
    
    # Training history
    train_losses = DEFAULT_FLOAT_TYPE[]
    val_losses = DEFAULT_FLOAT_TYPE[]
    val_r2_scores = DEFAULT_FLOAT_TYPE[]
    
    println("Starting training for up to $max_epochs epochs...")
    println("Early stopping: patience=$patience, min_delta=$min_delta")
    println("Batch size: $(train_dl.batchsize), Total batches per epoch: $(length(train_dl))")
    println("-" ^ 50)
    
    for epoch in 1:max_epochs
        # Train one epoch
        epoch_avg_loss, epoch_aux = train_epoch!(model, opt_state, train_dl, epoch, print_every; compute_loss)
        
        # Extract average valid count from auxiliary info
        valid_counts = [aux[:valid_count] for aux in epoch_aux if haskey(aux, :valid_count)]
        epoch_avg_valid = isempty(valid_counts) ? 0.0 : StatsBase.mean(valid_counts)
        
        # Evaluate validation metrics
        val_loss, individual_r2, aggregated_r2 = 
            evaluate_validation_metrics(model, val_dl, output_dim)
        
        # Store history
        push!(train_losses, epoch_avg_loss)
        push!(val_losses, val_loss)
        push!(val_r2_scores, aggregated_r2)
        
        # Print summary
        print_epoch_summary(epoch, epoch_avg_loss, val_loss, aggregated_r2, individual_r2, epoch_avg_valid; test_set=test_set)
        
        # Check early stopping
        best_val_loss, epochs_without_improvement, best_model_state, best_r2, should_stop = 
            check_early_stopping!(val_loss, best_val_loss, epochs_without_improvement, 
                                  best_model_state, model, min_delta, patience, 
                                  aggregated_r2, best_r2)
        
        println("-" ^ 50)
        
        should_stop && break
    end
    
    # Compile training statistics
    training_stats = Dict(
        :train_losses => train_losses,
        :val_losses => val_losses,
        :val_r2_scores => val_r2_scores,
        :best_val_loss => best_val_loss,
        :best_r2 => best_r2,
        :epochs_trained => length(train_losses),
        :converged => epochs_without_improvement < patience
    )
    
    return best_model_state, training_stats
end
