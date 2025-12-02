
"""Print epoch summary with training and validation metrics"""
function print_epoch_summary(epoch, train_loss, val_loss, aggregated_r2, individual_r2, avg_valid; test_set=false)
    valid_r2_scores = individual_r2[.!isnan.(individual_r2)]
    mean_individual_r2 = isempty(valid_r2_scores) ? NaN : mean(valid_r2_scores)
    
    println("Epoch $epoch Summary:")
    println("  Train Loss = $(round(train_loss, digits=6))")
    if test_set
        println("  Test Loss = $(round(val_loss, digits=6))")
    else
        println("  Val Loss = $(round(val_loss, digits=6))")
    end
    println("  Aggregated R² = $(round(aggregated_r2, digits=4))")
    println("  Individual R² Mean = $(round(mean_individual_r2, digits=4)) ($(sum(.!isnan.(individual_r2)))/$(length(individual_r2)) feature(s))")
    println("  Avg Valid Entries = $(round(avg_valid, digits=1))")
end