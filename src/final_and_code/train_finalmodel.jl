include("_helpers.jl")

"""
Train final model using combined train+val data, evaluate on test set.

# Arguments
- `raw_data`: Raw input data
- `create_model`: Function to create the model
- `seed`: Random seed (default: 1)
- `max_epochs`: Maximum training epochs for final model (default: 50)
- `patience`: Early stopping patience (default: 10)
- `print_every`: Print frequency (default: 100)
- `randomize_batchsize`: Whether to randomize batch size (default: true)
- `normalize_Y`: Whether to normalize labels (default: true)
- `normalization_method`: Normalization method (default: :zscore)
- `normalization_mode`: Normalization mode (default: :rowwise)
- `use_cuda`: Whether to use CUDA (default: true)
- `loss_fcn`: Loss function specification (default: (loss=Flux.mse, agg=StatsBase.mean))
- `model_kwargs...`: Additional model keyword arguments

# Returns
`(model, stats, train_stats, dl_train, dl_test)`
- `model`: Trained final model
- `stats`: Training statistics
- `train_stats`: Training setup statistics
- `dl_train`: Training dataloader (for optional processor training)
- `dl_test`: Test dataloader (for optional processor evaluation)

# Examples
```julia
# Train final model and get dataloaders
model, stats, train_stats, dl_train, dl_test = train_final_model(data, create_model; seed=42)

# Then optionally train processor using the same dataloaders
processor, proc_losses = train_code_processor(
    model, dl_train, MyModule.create_code_processor;
    arch_type=:mbconv,
    process_code_fn=MyModule.process_code_with_gradient,
    predict_from_code_fn=MyModule.predict_from_code
)
```
"""
function train_final_model(raw_data, create_model::Function; 
                          seed=1, max_epochs=50, patience=10, print_every=100,
                          randomize_batchsize=true, normalize_Y=true,
                          normalization_method=:zscore, normalization_mode=:rowwise,
                          use_cuda=true, loss_fcn=(loss=Flux.mse, agg=StatsBase.mean),
                          model_kwargs...)
    # Setup
    setup, batch_size = _prepare_final_model_setup(raw_data, create_model;
                                                    seed, randomize_batchsize, normalize_Y,
                                                    normalization_method, normalization_mode,
                                                    use_cuda, loss_fcn, model_kwargs...)
    
    # Create dataloaders
    dl_train, dl_test = _create_final_dataloaders(setup, batch_size, seed)
    
    if max_epochs == 0
        println("⚠️  max_epochs=0 specified, skipping training and returning untrained model.")
        return nothing, nothing, nothing, dl_train, dl_test
    end

    # Train final model
    trained_model, stats = _train_final_model!(setup, dl_train, dl_test;
                                               max_epochs, patience, print_every)
    
    return trained_model, stats, setup.train_stats, dl_train, dl_test
end


"""
Train final model from saved config (e.g., best trial from tuning).

# Arguments
- `raw_data`: Raw input data
- `create_model`: Function to create the model
- `config`: TrainingConfig from saved trial
- `max_epochs`: Maximum training epochs (default: 50)
- `patience`: Early stopping patience (default: 10)
- `print_every`: Print frequency (default: 100)
- `model_kwargs...`: Additional model keyword arguments

# Returns
`(model, stats, train_stats, dl_train, dl_test)`
"""
function train_final_model_from_config(raw_data, create_model::Function, config::TrainingConfig;
                                       max_epochs=50, patience=10, print_every=100,
                                       model_kwargs...)
    println("🎯 Training from config (seed=$(config.seed))...")
    train_final_model(raw_data, create_model; 
                     seed=config.seed, max_epochs, patience, print_every,
                     randomize_batchsize=config.randomize_batchsize, 
                     normalize_Y=config.normalize_Y,
                     normalization_method=config.normalization_method, 
                     normalization_mode=config.normalization_mode,
                     use_cuda=config.use_cuda, 
                     loss_fcn=config_to_loss_fcn(config),
                     model_kwargs...)
end

