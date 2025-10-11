function setup_model_and_training(
        data, 
        create_model::Union{Function,Nothing},
        batch_size::Int;
        normalize_Y=true,
        normalization_method=:zscore,
        normalization_mode=:rowwise,
        rng=Random.GLOBAL_RNG,
        use_cuda=true,
        create_new_model=true
        )
    """Common setup for model creation and data preparation"""
    
    # Create independent RNGs from the main RNG
    # This ensures reproducibility regardless of execution order
    model_rng = MersenneTwister(rand(rng, UInt))
    split_rng = MersenneTwister(rand(rng, UInt))
    
    # 1. Split data first (using split_rng)
    splits = train_val_test_split(data; _shuffle=true, rng=split_rng)
    
    # 2. Normalize labels if needed
    if normalize_Y
        train_stats = compute_normalization_stats(splits.train.Y; 
                method = normalization_method, mode = normalization_mode)
        train_Y = apply_normalization(splits.train.Y, train_stats)
        val_Y = apply_normalization(splits.val.Y, train_stats)
        test_Y = apply_normalization(splits.test.Y, train_stats)
    else
        train_stats = nothing
        train_Y, val_Y, test_Y = splits.train.Y, splits.val.Y, splits.test.Y
    end
    
    train_split = DataSplit(splits.train.X, train_Y, train_stats)
    val_split = DataSplit(splits.val.X, val_Y)
    test_split = DataSplit(splits.test.X, test_Y)
    processed_data = PreprocessedData(train_split, val_split, test_split)
    
    # 3. Optionally create model (using model_rng)
    if create_new_model
        if isnothing(create_model)
            error("create_model function must be provided when create_new_model=true")
        end
        
        result = _get_dims_and_model(data, create_model, batch_size; rng=model_rng, use_cuda=use_cuda)
        if result === nothing
            return nothing
        end
        
        return (
            model = result.model,
            optimizer_state = result.optimizer_state, 
            processed_data = processed_data,
            Ydim = result.Ydim,
            batch_size = batch_size,
        )
    else
        # Return data only (no model)
        Ydim = nothing
        try
            Ydim = data.Y_dim
        catch
            println("⚠️  Warning: Could not get Y_dim from data")
        end
        
        return (
            model = nothing,
            optimizer_state = nothing, 
            processed_data = processed_data,
            Ydim = Ydim,
            batch_size = batch_size,
        )
    end
end


function setup_model_and_training_final(
        data, 
        create_model::Union{Function,Nothing},
        batch_size::Int;
        normalize_Y=true,
        normalization_method=:zscore,
        normalization_mode=:rowwise,
        rng=Random.GLOBAL_RNG,
        use_cuda=true,
        create_new_model=true
        )
    """Setup for final training: combine train and val sets, return model, optimizer, processed data, Ydim."""
    
    # Create independent RNGs
    model_rng = MersenneTwister(rand(rng, UInt))
    split_rng = MersenneTwister(rand(rng, UInt))
    
    # 1. Split data first (using split_rng)
    splits = train_val_test_split(data; _shuffle=true, rng=split_rng)

    # 2. Combine train and val splits
    combined_X = cat(splits.train.X, splits.val.X, dims=ndims(splits.train.X))
    combined_Y = cat(splits.train.Y, splits.val.Y, dims=ndims(splits.train.Y))

    # 3. Normalize if needed
    if normalize_Y
        train_stats = compute_normalization_stats(combined_Y; 
                method = normalization_method, mode = normalization_mode)
        combined_Y = apply_normalization(combined_Y, train_stats)
        test_Y = apply_normalization(splits.test.Y, train_stats)
    else
        train_stats = nothing
        test_Y = splits.test.Y
    end

    train_split = DataSplit(combined_X, combined_Y, train_stats)
    test_split = DataSplit(splits.test.X, test_Y)
    processed_data = PreprocessedData(train_split, nothing, test_split)

    # 4. Optionally create model (using model_rng)
    if create_new_model
        if isnothing(create_model)
            error("create_model function must be provided when create_new_model=true")
        end
        
        result = _get_dims_and_model(data, create_model, batch_size; rng=model_rng, use_cuda=use_cuda)
        if result === nothing
            return nothing
        end

        return (
            model = result.model,
            optimizer_state = result.optimizer_state, 
            processed_data = processed_data,
            Ydim = result.Ydim,
            batch_size = batch_size,
            model_clone = deepcopy(result.model)
        )
    else
        # Return data only (no model)
        Ydim = nothing
        try
            Ydim = data.Y_dim
        catch
            println("⚠️  Warning: Could not get Y_dim from data")
        end
        
        return (
            model = nothing,
            optimizer_state = nothing, 
            processed_data = processed_data,
            Ydim = Ydim,
            batch_size = batch_size,
            model_clone = nothing
        )
    end
end

# Helper to get dims, create model, and optimizer
function _get_dims_and_model(data, create_model, batch_size; rng, use_cuda)
    Xdim, Ydim = nothing, nothing
    try
        Xdim, Ydim = data.X_dim, data.Y_dim
    catch
        println("⚠️  $(typeof(data)) is missing required fields:")
        println("    data.X_dim and data.Y_dim.")
        return nothing
    end
    m = create_model(Xdim, Ydim, batch_size; rng=rng, use_cuda=use_cuda)
    if isnothing(m)
        println("⚠️  Failed to create model with given hyperparameters.")
        return nothing
    end
    m = m |> FLUX_MODEL_FLOAT_FCN
    opt_state = Flux.setup(Flux.AdaBelief(), m)
    return (model=m, optimizer_state=opt_state, Ydim=Ydim)
end