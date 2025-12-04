# Internal helpers for hyperparameter tuning

"""Setup save file path with timestamp."""
_setup_save_file(folder) = (mkpath(folder); joinpath(folder, "results_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).csv"))

"""Save results if current trial is best."""
function _save_if_best!(df, file, r2, best_r2)
    isnothing(file) && return
    if r2 > best_r2
        try
            CSV.write(file, sort(df, :best_r2; rev=true))
            println("  🎉 NEW BEST R² = $(round(r2, digits=4))!")
        catch e
            println("  ⚠️  Save failed: $e")
        end
    else
        println("  📊 R² = $(round(r2, digits=4)) (best: $(round(best_r2, digits=4)))")
    end
end

"""
Run a single hyperparameter tuning trial.

# Returns: (r2, val_loss, num_params, model_state, model, batch_size)
"""
function _run_trial(trial, raw_data, create_model, randomize_batchsize, 
                    normalize_Y, normalization_method, normalization_mode,
                    use_cuda, loss_fcn, max_epochs, patience, print_every, model_kwargs)
    
    println("🔍 Trial $trial (seed: $trial)")
    
    rng = set_reproducible_seeds!(trial)
    batch_size = randomize_batchsize ? rand(rng, BATCH_SIZE_RANGE) : DEFAULT_BATCH_SIZE
    
    setup = setup_training(raw_data, create_model, batch_size;
                          normalize_Y, normalization_method, normalization_mode, 
                          rng, use_cuda, loss_fcn, model_kwargs...)
    
    if isnothing(setup)
        println("  ❌ Invalid setup, skipping...")
        return nothing
    end
    
    dl_train, dl_val, _ = obtain_data_loaders(setup.processed_data, batch_size;
                                               rng=MersenneTwister(rand(Random.GLOBAL_RNG, 1:typemax(Int))))
    
    model_state, stats = train_model(setup.model, setup.opt_state, dl_train, dl_val, setup.Ydim;
                                     max_epochs, patience, print_every, loss_fcn=setup.loss_fcn)
    
    r2, loss = stats[:best_r2], stats[:best_val_loss]
    num_params = sum(length, Flux.trainables(setup.model))
    
    return (r2=r2, val_loss=loss, num_params=num_params, model_state=model_state, 
            model=setup.model, batch_size=batch_size)
end

"""Save trial configuration to JSON file."""
function _save_trial_config(trial, batch_size, normalize_Y, normalization_method, 
                           normalization_mode, use_cuda, randomize_batchsize, 
                           loss_fcn, r2, loss, save_folder)
    config = TrainingConfig(
        seed=trial,
        batch_size=batch_size,
        normalize_Y=normalize_Y,
        normalization_method=normalization_method,
        normalization_mode=normalization_mode,
        use_cuda=use_cuda,
        randomize_batchsize=randomize_batchsize,
        loss_function=get_function_name(loss_fcn.loss),
        aggregation=get_function_name(loss_fcn.agg),
        best_r2=r2,
        val_loss=loss
    )
    save_trial_config(config, save_folder)
end

"""Update best model if current trial is better."""
function _update_best!(current_r2, best_r2, trial, batch_size, model, model_state)
    if current_r2 > best_r2
        best_model = Flux.deepcopy(model)
        Flux.loadmodel!(best_model, model_state)
        return current_r2, trial, batch_size, best_model
    end
    return best_r2, nothing, nothing, nothing
end

"""Print final tuning summary."""
function _print_summary(results, save_file, best_model, best_seed, best_r2, best_batch)
    if nrow(results) > 0
        sort!(results, :best_r2; rev=true)
        println("\n📈 Tuning complete! Results:\n", results)
        !isnothing(save_file) && CSV.write(save_file, results)
    end
    
    if !isnothing(best_model)
        println("\n🏆 Best: seed=$best_seed, R²=$(round(best_r2, digits=4)), batch=$best_batch")
        return (seed=best_seed, r2=best_r2, batch_size=best_batch)
    end
    return nothing
end
