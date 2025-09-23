
function setup_model_and_training(
        data, 
        create_model::Function,
        batch_size::Int;
        normalize_Y=true,
        normalization_method=:zscore,
        normalization_mode=:rowwise,
        rng=Random.GLOBAL_RNG,
        use_cuda=false
        )
    """Common setup for model creation and data preparation"""
    
    # 1. Get input and output dimensions
    Xdim, Ydim = nothing, nothing
    try
        Xdim, Ydim = data.X_dim, data.Y_dim
    catch
        println("⚠️  $(typeof(data)) is missing required fields:")
        println("    data.X_dim and data.Y_dim.")
        return nothing
    end

    # 2. Generate model
    m = nothing
    m = create_model(Xdim, Ydim, batch_size; rng=rng, use_cuda=use_cuda)
    if isnothing(m)
        println("⚠️  Failed to create model with given hyperparameters.")
        return nothing
    end

    opt_state = Flux.setup(Flux.AdaBelief(), m) # TODO: make optimizer configurable

    # 3. split the data, normalize, and create data loaders
    # 3-1 Split data first (no processing)
    splits = train_val_test_split(data; _shuffle=true, rng=rng)
    splits.val.Y |> size |> println
    # 3-2 Normalize labels if needed
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
        
    return (
        model = m,
        optimizer_state = opt_state, 
        processed_data = processed_data,
        Ydim = Ydim
    )
end
