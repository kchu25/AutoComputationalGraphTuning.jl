# Core setup: split data, normalize, optionally create model
function setup_training(data, create_model, batch_size; combine_train_val=false, 
                       normalize_Y=true, normalization_method=:zscore, normalization_mode=:rowwise,
                       clip_quantiles=(0.00001, 0.99999),
                       rng=Random.GLOBAL_RNG, use_cuda=true, loss_spec=(loss=Flux.mse, agg=StatsBase.mean),
                       model_kwargs...)
    model_rng, split_rng = MersenneTwister(rand(rng, UInt)), MersenneTwister(rand(rng, UInt))
    splits, splits_indices = train_val_test_split(data; _shuffle=true, rng=split_rng)
    
    # Prepare data based on mode
    if combine_train_val
        X = cat(splits.train.X, splits.val.X, dims=ndims(splits.train.X))
        Y = cat(splits.train.Y, splits.val.Y, dims=ndims(splits.train.Y))
        train_stats, Y_norm, test_Y = if normalize_Y
            stats = compute_normalization_stats(Y; method=normalization_method, mode=normalization_mode, clip_quantiles=clip_quantiles)
            (stats, apply_normalization(Y, stats), apply_normalization(splits.test.Y, stats))
        else
            (nothing, Y, splits.test.Y)
        end
        processed_data = PreprocessedData(DataSplit(X, Y_norm, train_stats), nothing, DataSplit(splits.test.X, test_Y))
        # Merge train and val indices so split_indices.train reflects the combined set
        splits_indices = (train=vcat(splits_indices.train, splits_indices.val), val=Int[], test=splits_indices.test)
    else
        train_stats, train_Y, val_Y, test_Y = if normalize_Y
            stats = compute_normalization_stats(splits.train.Y; method=normalization_method, mode=normalization_mode, clip_quantiles=clip_quantiles)
            (stats, apply_normalization(splits.train.Y, stats), 
             apply_normalization(splits.val.Y, stats), apply_normalization(splits.test.Y, stats))
        else
            (nothing, splits.train.Y, splits.val.Y, splits.test.Y)
        end
        processed_data = PreprocessedData(DataSplit(splits.train.X, train_Y, train_stats),
                                         DataSplit(splits.val.X, val_Y), DataSplit(splits.test.X, test_Y))
    end
    
    # Create model if requested
    model, opt_state, Ydim = if !isnothing(create_model)
        try
            Xdim, Ydim = data.X_dim, data.Y_dim
            m = create_model(Xdim, Ydim, batch_size; rng=model_rng, use_cuda=use_cuda, model_kwargs...) |> FLUX_MODEL_FLOAT_FCN
            isnothing(m) && return nothing
            (m, Flux.setup(Flux.AdaBelief(), m), Ydim)
        catch e
            println("⚠️  Failed to create model: $e")
            println(stacktrace(catch_backtrace()))
            println("⚠️  Failed to create model")
            return nothing
        end
    else
        (nothing, nothing, data.Y_dim)
    end
    
    (model=model, opt_state=opt_state, processed_data=processed_data, Ydim=Ydim, batch_size=batch_size,
     model_clone=isnothing(model) ? nothing : deepcopy(model), train_stats=train_stats,
     compiled_loss=compile_loss(loss_spec), split_indices=splits_indices)
end

# Backward compatibility wrappers
setup_model_and_training(args...; kwargs...) = setup_training(args...; combine_train_val=false, kwargs...)
setup_model_and_training_final(args...; kwargs...) = setup_training(args...; combine_train_val=true, kwargs...)