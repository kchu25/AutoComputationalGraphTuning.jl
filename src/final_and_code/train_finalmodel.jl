include("_helpers.jl")

"""
Train final model using combined train+val data, evaluate on test set.

# Arguments
- `raw_data`: Raw input data
- `create_model`: Function to create the model
- `seed`, `max_epochs`, `patience`, `print_every`: Training hyperparameters
- `randomize_batchsize`, `normalize_Y`, `normalization_method`, `normalization_mode`: Data config
- `use_cuda`, `loss_fcn`: Compute settings
- `model_kwargs...`: Additional model arguments

# Returns
`(model, stats, train_stats, dl_train, dl_test)` where dataloaders can be reused for processor training

# Example
```julia
model, stats, train_stats, dl_train, dl_test = train_final_model(data, create_model; seed=42)
processor, _ = train_code_processor(model, dl_train, proc_wrap)
```
"""
function train_final_model(raw_data, create_model::Function; 
                          seed=1, max_epochs=50, patience=10, print_every=100,
                          randomize_batchsize=true, normalize_Y=true,
                          normalization_method=:zscore, normalization_mode=:rowwise,
                          use_cuda=true, 
                          loss_fcn=(loss=Flux.mse, agg=StatsBase.mean),
                          model_kwargs...)
    
    # Setup and create dataloaders
    setup, batch_size = _prepare_final_model_setup(raw_data, create_model; seed, randomize_batchsize,
                                                    normalize_Y, normalization_method, normalization_mode,
                                                    use_cuda, loss_fcn, model_kwargs...)
    dl_train, dl_test = _create_final_dataloaders(setup, batch_size, seed)
    
    # Skip training if max_epochs=0
    if max_epochs == 0
        println("⚠️  max_epochs=0; return only the dataloaders for final model training.")
        return nothing, nothing, nothing, dl_train, dl_test, setup.split_indices
    end

    # Train and return
    trained_model, stats = _train_final_model!(setup, dl_train, dl_test; max_epochs, patience, print_every)
    return trained_model, stats, setup.train_stats, dl_train, dl_test, setup.split_indices
end

"""
Train final model from saved config (e.g., best trial from tuning).

# Returns
`(model, stats, train_stats, dl_train, dl_test)`
"""
function train_final_model_from_config(raw_data, create_model::Function, config::TrainingConfig, trc; max_epochs=50, patience=10, print_every=100, model_kwargs...)
    println("🎯 Training from config (seed=$(config.seed))...")

    loss_fcn = trc.loss_fcn;
    seed = trc.seed;
    normalization_method = trc.normalization_method;

    train_final_model(raw_data, create_model; 
        seed=seed, max_epochs, patience, print_every,
        randomize_batchsize=config.randomize_batchsize, 
        normalize_Y=config.normalize_Y,
        normalization_method=normalization_method, 
        normalization_mode=config.normalization_mode,
        use_cuda=config.use_cuda, loss_fcn=loss_fcn, 
        model_kwargs...)
end

