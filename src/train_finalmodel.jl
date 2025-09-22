

function train_final_model(
    raw_data, 
    create_model::Function; 
    seed=1,
    max_epochs=50,
    patience=10,
    print_every=100,
    randomize_batchsize = true
    )
    """Train final model using training + validation data"""

    rng_global = set_reproducible_seeds!(seed)

    # Setup model and data
    setup = setup_model_and_training(raw_data, create_model, randomize_batchsize; 
        rng=rng_global)

    isnothing(setup) && error("Invalid hyperparameters for final model training")
    
    # Combine train and validation data
    combined_tensor = cat(setup.processed_data.train.tensor, setup.processed_data.val.tensor, 
        dims=ndims(setup.processed_data.train.tensor))
    combined_labels = cat(setup.processed_data.train.labels, setup.processed_data.val.labels, 
        dims=ndims(setup.processed_data.train.labels))
    
    # Create combined training dataloader
    dl_combined = Flux.DataLoader(
        (combined_tensor, combined_labels),
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

    println("🎯 Training final model with combined train+validation data...")
    
    # Train final model
    final_model_state, stats = train_model(
        setup.model, setup.optimizer_state, dl_combined, dl_test, setup.feature_counts;
        max_epochs=max_epochs, patience=patience, print_every=print_every, 
        test_set = true
    )

    return final_model_state, stats
end