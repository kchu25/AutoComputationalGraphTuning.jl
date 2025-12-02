# Helpers
_setup_save_file(folder) = (mkpath(folder); joinpath(folder, "results_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).csv"))

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
Tune hyperparameters across multiple trials with different seeds.

# Returns: (results_df, best_model, best_info)
- results_df: DataFrame with all trial results
- best_model: Model with highest validation R²
- best_info: NamedTuple (seed, r2, batch_size) of best trial
"""
function tune_hyperparameters(raw_data, model_module::Module;
                              randomize_batchsize=true, max_epochs=50, patience=5,
                              trial_number_start=1, n_trials=100,
                              normalize_Y=true, normalization_method=:zscore, normalization_mode=:rowwise,
                              print_every=100, save_folder=nothing, use_cuda=true,
                              loss_fcn=(loss=Flux.mse, agg=StatsBase.mean), model_kwargs...)
    
    results = DataFrame(seed=Int[], best_r2=DEFAULT_FLOAT_TYPE[], val_loss=DEFAULT_FLOAT_TYPE[], num_params=Int[])
    best_r2, best_model, best_seed, best_batch = -Inf, nothing, nothing, nothing
    save_file = isnothing(save_folder) ? nothing : _setup_save_file(save_folder)
    
    for trial in trial_number_start:(trial_number_start+n_trials-1)
        println("🔍 Trial $trial (seed: $trial)")
        
        rng = set_reproducible_seeds!(trial)
        batch_size = randomize_batchsize ? rand(rng, BATCH_SIZE_RANGE) : DEFAULT_BATCH_SIZE
        
        setup = setup_training(raw_data, model_module.create_model, batch_size;
                              normalize_Y, normalization_method, normalization_mode, rng, use_cuda, loss_fcn, model_kwargs...)
        
        if isnothing(setup)
            println("  ❌ Invalid setup, skipping...")
            continue
        end
        
        dl_train, dl_val, _ = obtain_data_loaders(setup.processed_data, batch_size;
                                                   rng=MersenneTwister(rand(Random.GLOBAL_RNG, 1:typemax(Int))))
        
        model_state, stats = train_model(setup.model, setup.opt_state, dl_train, dl_val, setup.Ydim;
                                         max_epochs, patience, print_every, loss_fcn=setup.loss_fcn)
        
        r2, loss = stats[:best_r2], stats[:best_val_loss]
        num_params = sum(length, Flux.trainables(setup.model))
        push!(results, (trial, r2, loss, num_params))
        
        # Save trial config to JSON
        if !isnothing(save_folder)
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
        
        if r2 > best_r2
            best_r2, best_seed, best_batch = r2, trial, batch_size
            best_model = Flux.deepcopy(setup.model)
            Flux.loadmodel!(best_model, model_state)
        end
        
        _save_if_best!(results, save_file, r2, best_r2)
    end
    
    if nrow(results) > 0
        sort!(results, :best_r2; rev=true)
        println("\n📈 Tuning complete! Results:\n", results)
        !isnothing(save_file) && CSV.write(save_file, results)
    end
    
    best_info = isnothing(best_model) ? nothing : (seed=best_seed, r2=best_r2, batch_size=best_batch)
    !isnothing(best_model) && println("\n🏆 Best: seed=$best_seed, R²=$(round(best_r2, digits=4)), batch=$best_batch")
    
    results, best_model, best_info
end
