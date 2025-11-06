# Helper: Setup save file and folder
function _setup_save_file(save_folder)
    mkpath(save_folder)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    save_file = joinpath(save_folder, "hyperparameter_results_$(timestamp).csv")
    println("📂 Results will be saved to: $save_file")
    return save_file
end

# Helper: Setup trial with RNG and model
function _setup_trial(raw_data, create_model, trial_number; 
                      randomize_batchsize, normalize_Y, 
                      normalization_method, normalization_mode, use_cuda, create_new_model, 
                      loss_fcn=(loss=Flux.mse, agg=StatsBase.mean))
    rng_global = set_reproducible_seeds!(trial_number)
    batch_size = randomize_batchsize ? rand(rng_global, BATCH_SIZE_RANGE) : DEFAULT_BATCH_SIZE
    
    setup = setup_model_and_training(
        raw_data, 
        create_model,
        batch_size;
        normalize_Y=normalize_Y,
        normalization_method=normalization_method,
        normalization_mode=normalization_mode, 
        rng=rng_global,
        use_cuda=use_cuda,
        create_new_model=create_new_model,
        loss_fcn=loss_fcn
    )
    
    return rng_global, setup
end

# Helper: Save results if new best R²
function _maybe_save_results!(results_df, save_file, current_r2, best_r2_so_far)
    if isnothing(save_file)
        return
    end
    if current_r2 > best_r2_so_far
        try
            sorted_df = sort(results_df, :best_r2; rev=true)
            CSV.write(save_file, sorted_df)
            println("  🎉 NEW BEST R² = $(round(current_r2, digits=4))! Results saved ($(nrow(results_df)) trials completed)")
        catch e
            println("  ⚠️  Warning: Could not save results - $e")
        end
    else
        println("  📊 R² = $(round(current_r2, digits=4)) (best so far: $(round(best_r2_so_far, digits=4)))")
    end
end

# Helper: Print and save final results
function _print_and_save_final_results!(results_df, save_file, loss_fcn)
    if nrow(results_df) > 0
        sort!(results_df, :best_r2; rev=true)
        println("\n📈 Hyperparameter tuning results:")
        println("Loss configuration: $(loss_fcn.loss) with $(loss_fcn.agg) aggregation")
        println(results_df)
        if !isnothing(save_file)
            try
                CSV.write(save_file, results_df)
                println("📂 Final results saved to: $save_file")
            catch e
                println("⚠️  Warning: Could not save final results - $e")
            end
        end
    else
        println("\n⚠️  No successful trials - all hyperparameters were invalid!")
    end
end

function save_kwargs_to_file(kwargs::NamedTuple, filename::AbstractString)
    open(filename, "w") do io
        for (k, v) in pairs(kwargs)
            println(io, "$k = $v")
        end
    end
end

"""
    tune_hyperparameters(raw_data::SEQ2EXP_Dataset; seq_type=:Nucleotide, randomize_batchsize=false, max_epochs=50, patience=5, trial_number_start=1, n_trials=10, suppress_warnings=false, save_folder=nothing)

Tune hyperparameters for a model using train/validation split. Runs multiple trials, each with a different random seed, and tracks the best validation R² score and loss. Optionally saves results to CSV.

# Arguments
- `raw_data::SEQ2EXP_Dataset`: The dataset to use for training/validation
- `seq_type`: Sequence type (default: `:Nucleotide`)
- `randomize_batchsize`: Whether to randomize batch size per trial
- `max_epochs`: Maximum epochs per trial
- `patience`: Early stopping patience
- `trial_number_start`: Starting trial number/seed
- `n_trials`: Number of trials to run
- `suppress_warnings`: Suppress warnings during setup
- `save_folder`: If provided, results are saved to this folder as CSV

# Returns
- `results_df`: DataFrame with trial, best R², and validation loss for each trial
- `best_model`: The model with the best validation R² across all trials
- `best_trial_info`: NamedTuple with information about the best trial (seed, r2, loss, batch_size)
"""
function tune_hyperparameters(
    raw_data, 
    create_model::Function; 
    randomize_batchsize = true,
    max_epochs=50,
    patience=5,
    trial_number_start=1,
    n_trials=100,
    normalize_Y=true,
    normalization_method=:zscore,
    normalization_mode=:rowwise,
    print_every=100,
    save_folder=nothing,
    use_cuda=true,
    create_new_model=true,
    loss_fcn=(loss=Flux.mse, agg=StatsBase.mean)
    )

    results_df = DataFrame(
        seed = Int[], 
        best_r2 = DEFAULT_FLOAT_TYPE[], 
        val_loss = DEFAULT_FLOAT_TYPE[],
        loss_function = String[],
        aggregation = String[]
        )

    best_r2_so_far = -Inf
    best_model_state = nothing
    best_model_clone = nothing
    best_trial_seed = nothing
    best_trial_batch_size = nothing
    save_file = isnothing(save_folder) ? nothing : _setup_save_file(save_folder)

    if !isnothing(save_folder)
        save_kwargs_to_file((
            randomize_batchsize=randomize_batchsize,
            max_epochs=max_epochs,
            patience=patience,
            trial_number_start=trial_number_start,
            n_trials=n_trials,
            normalize_Y=normalize_Y,
            normalization_method=normalization_method,
            normalization_mode=normalization_mode,
            print_every=print_every,
            use_cuda=use_cuda,
            loss_fcn=loss_fcn
        ), joinpath(save_folder, "tuning_args.txt")) 
    end

    for trial_number in trial_number_start:(trial_number_start+n_trials-1)
        
        println("🔍 Hyperparameter trial $trial_number (seed: $trial_number)")

        rng_global, setup = _setup_trial(raw_data, create_model, trial_number; 
                                         randomize_batchsize=randomize_batchsize,
                                         normalize_Y=normalize_Y,
                                         normalization_method=normalization_method,
                                         normalization_mode=normalization_mode,
                                         use_cuda=use_cuda,
                                         create_new_model=create_new_model,
                                         loss_fcn=loss_fcn)

        if isnothing(setup)
            println("  ❌ Invalid setup, skipping...")
            continue
        end

        dl_train, dl_val, _ = obtain_data_loaders(
                setup.processed_data, 
                setup.batch_size; 
                rng = MersenneTwister(rand(Random.GLOBAL_RNG, 1:typemax(Int)))
                # use this because Flux.DataLoader requires an integer seed
                )

        model_state, stats = train_model(setup.model, 
                               setup.optimizer_state, 
                               dl_train, dl_val, 
                               setup.Ydim;
                               max_epochs=max_epochs, 
                               patience=patience, 
                               print_every=print_every,
                               loss_fcn=setup.loss_fcn
                               )
        current_r2, val_loss = stats[:best_r2], stats[:best_val_loss]
        
        # Create and save trial configuration
        trial_config = TrainingConfig(
            seed = trial_number,
            normalize_Y = normalize_Y,
            normalization_method = normalization_method,
            normalization_mode = normalization_mode,
            use_cuda = use_cuda,
            randomize_batchsize = randomize_batchsize,
            batch_size = setup.batch_size,
            loss_function = get_function_name(loss_fcn.loss),
            aggregation = get_function_name(loss_fcn.agg),
            best_r2 = current_r2,
            val_loss = val_loss
        )
        
        if !isnothing(save_folder)
            save_trial_config(trial_config, save_folder)
        end
        
        # Convert loss function and aggregation to string representations
        loss_str = get_function_name(loss_fcn.loss)
        agg_str = get_function_name(loss_fcn.agg)
        push!(results_df, (trial_number, current_r2, val_loss, loss_str, agg_str))
        
        # Track best model across all trials
        if current_r2 > best_r2_so_far
            best_model_state = model_state
            best_model_clone = Flux.deepcopy(setup.model)
            Flux.loadmodel!(best_model_clone, model_state)
            best_trial_seed = trial_number
            best_trial_batch_size = setup.batch_size
        end
        
        _maybe_save_results!(results_df, save_file, current_r2, best_r2_so_far)
        best_r2_so_far = max(best_r2_so_far, current_r2)
    end
    _print_and_save_final_results!(results_df, save_file, loss_fcn)
    
    # Create best trial info
    best_trial_info = if !isnothing(best_model_clone)
        (
            seed = best_trial_seed,
            r2 = best_r2_so_far,
            batch_size = best_trial_batch_size
        )
    else
        nothing
    end
    
    if !isnothing(best_model_clone)
        println("\n🏆 Best model found:")
        println("   Seed: $best_trial_seed")
        println("   R²: $(round(best_r2_so_far, digits=4))")
        println("   Batch size: $best_trial_batch_size")
    end
    
    return results_df, best_model_clone, best_trial_info
end
