
"""Train single batch and return loss and statistics"""
function train_batch!(model, opt_state, seq, labels; loss_fcn=masked_mse)
    seq, labels = seq |> cu, labels |> cu
    nan_mask = .!isnan.(labels)

    # Compute loss and gradients
    loss, gs = Flux.withgradient(model) do x
        loss_fcn(x(seq), labels, nan_mask)
    end
    
    # Update model parameters
    Flux.update!(opt_state, model, gs[1])
    
    return loss, sum(nan_mask)
end

"""Train single epoch and return epoch statistics"""
function train_epoch!(model, opt_state, dataloader, epoch, print_every; loss_fcn=masked_mse)
    epoch_losses = DEFAULT_FLOAT_TYPE[]
    epoch_valid_counts = Int[]

    for (batch_idx, (seq, labels)) in enumerate(dataloader)
        loss, valid_count = train_batch!(model, opt_state, seq, labels; loss_fcn=loss_fcn)

        push!(epoch_losses, loss)
        push!(epoch_valid_counts, valid_count)
        
        # Print progress
        if batch_idx % print_every == 0
            avg_loss = mean(epoch_losses)
            println("Epoch $epoch, Batch $batch_idx: Loss = $(round(loss, digits=6)), " * 
                   "Avg Loss = $(round(avg_loss, digits=6)), " *
                   "Valid entries: $valid_count/$(length(.!isnan.(labels)))")
        end
    end
    
    return mean(epoch_losses), mean(epoch_valid_counts)
end

# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

"""
    train_model(model, opt_state, train_dl, val_dl, output_dim; 
                max_epochs=50, patience=10, min_delta=1e-4, print_every=100)

Train a model with early stopping and return the best model state and training stats.

# Returns
- `best_model_state`: State dict of the best model (lowest validation loss)
- `training_stats`: Dict with training history and final metrics
"""
function train_model(model, opt_state, train_dl, val_dl, output_dim;
                     max_epochs=50, 
                     patience=10, 
                     min_delta=1e-4, 
                     print_every=100, 
                     test_set=false
                     )
    
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
        epoch_avg_loss, epoch_avg_valid = train_epoch!(model, opt_state, train_dl, epoch, print_every)
        
        # Evaluate validation metrics
        val_loss, individual_r2, aggregated_r2 = 
            CNN.evaluate_validation_metrics(model, val_dl, output_dim)
        
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