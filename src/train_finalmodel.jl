"""
    get_data_combined_and_data_test(raw_data; create_model, seed, randomize_batchsize, normalize_Y, normalization_method, normalization_mode, use_cuda)

Prepare combined training data (train+val) and test data loaders for final model training.
"""

# Helper: Setup final model trial with RNG and model
function _setup_final_trial(raw_data, create_model, seed;
                            randomize_batchsize, normalize_Y,
                            normalization_method, normalization_mode, use_cuda, 
                            loss_fcn=(loss=Flux.mse, agg=StatsBase.mean))
    rng_global = set_reproducible_seeds!(seed)
    batch_size = randomize_batchsize ? rand(rng_global, BATCH_SIZE_RANGE) : DEFAULT_BATCH_SIZE
    
    setup = setup_model_and_training_final(
        raw_data, 
        create_model, 
        batch_size;
        normalize_Y=normalize_Y,
        normalization_method=normalization_method,
        normalization_mode=normalization_mode,
        rng=rng_global,
        use_cuda=use_cuda,
        loss_fcn=loss_fcn
    )
    
    return rng_global, setup
end

function get_data_combined_and_data_test(
    raw_data,
    create_model::Function; 
    seed=1,
    randomize_batchsize = true,
    normalize_Y=true,
    normalization_method=:zscore,
    normalization_mode=:rowwise,
    use_cuda=true,
    loss_fcn=(loss=Flux.mse, agg=StatsBase.mean)
)
    rng_global, setup = _setup_final_trial(
        raw_data, create_model, seed;
        randomize_batchsize=randomize_batchsize,
        normalize_Y=normalize_Y,
        normalization_method=normalization_method,
        normalization_mode=normalization_mode,
        use_cuda=use_cuda,
        loss_fcn=loss_fcn
    )

    isnothing(setup) && error("Yo man. Invalid hyperparameters for final model training")

    # Create combined training dataloader
    dl_combined = Flux.DataLoader(
        (setup.processed_data.train.tensor, setup.processed_data.train.labels), # this is train + val
        batchsize = setup.batch_size,
        shuffle = true,
        partial = false,
        rng = MersenneTwister(seed)
    )

    # Create test dataloader
    dl_test = Flux.DataLoader(
        (setup.processed_data.test.tensor, setup.processed_data.test.labels),
        batchsize = setup.batch_size,
        shuffle = false,
        partial = false,
        rng = MersenneTwister(seed)
    )
    return setup, dl_combined, dl_test
end


function train_final_model(
    raw_data, 
    create_model::Function; 
    seed=1,
    max_epochs=50,
    patience=10,
    print_every=100,
    randomize_batchsize = true,
    normalize_Y=true,
    normalization_method=:zscore,
    normalization_mode=:rowwise,
    use_cuda=true,
    loss_fcn=(loss=Flux.mse, agg=StatsBase.mean)
    )
    """Train final model using training + validation data (combined)"""
    setup, dl_combined, dl_test = get_data_combined_and_data_test(
        raw_data, 
        create_model; 
        seed=seed,
        randomize_batchsize=randomize_batchsize,
        normalize_Y=normalize_Y,
        normalization_method=normalization_method,
        normalization_mode=normalization_mode,
        use_cuda=use_cuda,
        loss_fcn=loss_fcn
    )

    println("🎯 Training final model with combined train+validation data...")

    # Train final model
    best_model_state, stats = train_model(
        setup.model, setup.optimizer_state, dl_combined, dl_test, setup.Ydim;
        max_epochs=max_epochs, patience=patience, print_every=print_every, 
        test_set = true, loss_fcn=setup.loss_fcn
    )

    Flux.loadmodel!(setup.model_clone, best_model_state) # load the best model state
    return setup.model_clone, stats, setup.train_stats
    # model, stats (performances), and train_stats (scaling utils)
end


"""
    train_final_model_from_config(raw_data, create_model::Function, config::TrainingConfig; 
                                   max_epochs=50, patience=10, print_every=100)

Train final model using configuration loaded from a JSON file (e.g., from tuning results).
This ensures the same settings used during the best trial are applied to final training.

# Arguments
- `raw_data`: The dataset to use for training
- `create_model`: Function to create the model architecture
- `config`: TrainingConfig loaded from JSON
- `max_epochs`: Maximum epochs for final training
- `patience`: Early stopping patience
- `print_every`: Print progress every N epochs

# Returns
- `model`: Trained model
- `stats`: Training statistics

# Example
```julia
# Load the best trial configuration from tuning
config = load_best_trial_config("results/tuning_run_2024-01-01")

# Train final model with the same settings
model, stats = train_final_model_from_config(data, create_model, config)
```
"""
function train_final_model_from_config(
    raw_data,
    create_model::Function,
    config::TrainingConfig;
    max_epochs=50,
    patience=10,
    print_every=100
)
    println("🎯 Training final model from config (seed=$(config.seed))...")
    
    loss_fcn = config_to_loss_fcn(config)
    
    return train_final_model(
        raw_data,
        create_model;
        seed=config.seed,
        max_epochs=max_epochs,
        patience=patience,
        print_every=print_every,
        randomize_batchsize=config.randomize_batchsize,
        normalize_Y=config.normalize_Y,
        normalization_method=config.normalization_method,
        normalization_mode=config.normalization_mode,
        use_cuda=config.use_cuda,
        loss_fcn=loss_fcn
    )
end


