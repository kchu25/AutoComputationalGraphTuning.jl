"""Check early stopping condition and update best model state"""
function check_early_stopping!(val_loss, best_val_loss, epochs_without_improvement, 
                               best_model_state, model, min_delta, patience, 
                               aggregated_r2, best_r2)
    if val_loss < best_val_loss - min_delta && aggregated_r2 > 0
        best_val_loss = val_loss
        best_r2 = aggregated_r2
        epochs_without_improvement = 0
        best_model_state = deepcopy(Flux.state(model))
        println("✓ New best validation loss: $(round(best_val_loss, digits=6)), R² = $(round(best_r2, digits=4))")
        return best_val_loss, epochs_without_improvement, best_model_state, best_r2, false
    else
        epochs_without_improvement += 1
        println("⚠ No improvement for $epochs_without_improvement epoch(s)")
        
        should_stop = epochs_without_improvement >= patience || aggregated_r2 ≤ 0
        if should_stop
            println("Early stopping triggered!")
            println("Best validation loss: $(round(best_val_loss, digits=6)), Best R² = $(round(best_r2, digits=4))")
        end
        
        return best_val_loss, epochs_without_improvement, best_model_state, best_r2, should_stop
    end
end
