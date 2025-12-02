"""
Train final model using combined train+val data, evaluate on test set.

# Returns: (model, stats, train_stats)
"""
function train_final_model(raw_data, model_module::Module; 
                          seed=1, max_epochs=50, patience=10, print_every=100,
                          randomize_batchsize=true, normalize_Y=true,
                          normalization_method=:zscore, normalization_mode=:rowwise,
                          use_cuda=true, loss_fcn=(loss=Flux.mse, agg=StatsBase.mean),
                          model_kwargs...)
    rng = set_reproducible_seeds!(seed)
    batch_size = randomize_batchsize ? rand(rng, BATCH_SIZE_RANGE) : DEFAULT_BATCH_SIZE
    
    setup = setup_training(raw_data, model_module.create_model, batch_size; combine_train_val=true,
                          normalize_Y, normalization_method, normalization_mode, rng, use_cuda, loss_fcn, model_kwargs...)
    isnothing(setup) && error("Invalid hyperparameters for final model training")
    
    dl_train = Flux.DataLoader((setup.processed_data.train.tensor, setup.processed_data.train.labels),
                               batchsize=batch_size, shuffle=true, partial=false, rng=MersenneTwister(seed))
    dl_test = Flux.DataLoader((setup.processed_data.test.tensor, setup.processed_data.test.labels),
                              batchsize=batch_size, shuffle=false, partial=false)
    
    println("🎯 Training final model (train+val combined)...")
    best_state, stats = train_model(setup.model, setup.opt_state, dl_train, dl_test, setup.Ydim;
                                    max_epochs, patience, print_every, test_set=true, loss_fcn=setup.loss_fcn)
    
    Flux.loadmodel!(setup.model_clone, best_state)
    setup.model_clone, stats, setup.train_stats
end

"""Train final model from saved config (e.g., best trial from tuning)"""
function train_final_model_from_config(raw_data, model_module::Module, config::TrainingConfig;
                                       max_epochs=50, patience=10, print_every=100, model_kwargs...)
    println("🎯 Training from config (seed=$(config.seed))...")
    train_final_model(raw_data, model_module; seed=config.seed, max_epochs, patience, print_every,
                     randomize_batchsize=config.randomize_batchsize, normalize_Y=config.normalize_Y,
                     normalization_method=config.normalization_method, normalization_mode=config.normalization_mode,
                     use_cuda=config.use_cuda, loss_fcn=config_to_loss_fcn(config), model_kwargs...)
end


