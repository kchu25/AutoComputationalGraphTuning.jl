include("_helpers.jl")

"""
Tune hyperparameters across multiple trials with different seeds.

# Returns: (results_df, best_model, best_info)
- results_df: DataFrame with all trial results
- best_model: Model with highest validation R²
- best_info: NamedTuple (seed, r2, batch_size) of best trial
"""
function tune_hyperparameters(raw_data, create_model::Function;
                              randomize_batchsize=true, max_epochs=50, patience=5,
                              trial_number_start=1, n_trials=100,
                              normalize_Y=true, normalization_method=:zscore, normalization_mode=:rowwise,
                              print_every=100, save_folder=nothing, use_cuda=true,
                              loss_spec=(loss=Flux.mse, agg=StatsBase.mean), model_kwargs...)
    
    results = DataFrame(seed=Int[], best_r2=DEFAULT_FLOAT_TYPE[], val_loss=DEFAULT_FLOAT_TYPE[], num_params=Int[])
    best_r2, best_model, best_seed, best_batch = -Inf, nothing, nothing, nothing
    save_file = isnothing(save_folder) ? nothing : _setup_save_file(save_folder)
    
    for trial in trial_number_start:(trial_number_start+n_trials-1)
        # Run trial
        result = _run_trial(trial, raw_data, create_model, randomize_batchsize,
                           normalize_Y, normalization_method, normalization_mode,
                           use_cuda, loss_spec, max_epochs, patience, print_every, model_kwargs)
        
        isnothing(result) && continue
        
        # Record results
        push!(results, (trial, result.r2, result.val_loss, result.num_params))
        
        # Save trial config
        !isnothing(save_folder) && _save_trial_config(trial, result.batch_size, normalize_Y, 
                                                       normalization_method, normalization_mode, 
                                                       use_cuda, randomize_batchsize, loss_spec, 
                                                       result.r2, result.val_loss, save_folder)
        
        # Update best model
        new_best_r2, new_seed, new_batch, new_model = _update_best!(result.r2, best_r2, trial, 
                                                                     result.batch_size, result.model, 
                                                                     result.model_state)
        if !isnothing(new_seed)
            best_r2, best_seed, best_batch, best_model = new_best_r2, new_seed, new_batch, new_model
        end
        
        # Save if best
        _save_if_best!(results, save_file, result.r2, best_r2)
    end
    
    # Print summary and return
    best_info = _print_summary(results, save_file, best_model, best_seed, best_r2, best_batch)
    
    return results, best_model, best_info
end
