
"""Print epoch summary with training and validation metrics"""
function print_epoch_summary(epoch, train_loss, val_loss, aggregated_r2, individual_r2, avg_valid; test_set=false)
    valid_r2_scores = individual_r2[.!isnan.(individual_r2)]
    mean_individual_r2 = isempty(valid_r2_scores) ? NaN : mean(valid_r2_scores)
    
    vprintln(VERBOSITY_VERBOSE, "Epoch $epoch Summary:")
    vprintln(VERBOSITY_VERBOSE, "  Train Loss = $(round(train_loss, digits=6))")
    if test_set
        vprintln(VERBOSITY_VERBOSE, "  Test Loss = $(round(val_loss, digits=6))")
    else
        vprintln(VERBOSITY_VERBOSE, "  Val Loss = $(round(val_loss, digits=6))")
    end
    vprintln(VERBOSITY_VERBOSE, "  Aggregated R² = $(round(aggregated_r2, digits=4))")
    vprintln(VERBOSITY_VERBOSE, "  Individual R² Mean = $(round(mean_individual_r2, digits=4)) ($(sum(.!isnan.(individual_r2)))/$(length(individual_r2)) feature(s))")
    vprintln(VERBOSITY_VERBOSE, "  Avg Valid Entries = $(round(avg_valid, digits=1))")
end