"""Set all random seeds for reproducible results"""
function set_reproducible_seeds!(seed::Int = 42)
    # Julia's built-in random number generator
    Random.seed!(seed)
    
    # CUDA random number generator (if using GPU)
    if CUDA.functional()
        CUDA.seed!(seed)
    end
    
    println("🌱 Random seeds set to $seed for reproducibility")
    return Random.GLOBAL_RNG
end

function obtain_data_loaders(processed_data, batch_size; rng=nothing)
    dl_train = Flux.DataLoader(
        (processed_data.train.tensor, processed_data.train.labels),
        batchsize = batch_size,
        shuffle = true,
        partial = false,
        rng = isnothing(rng) ? Random.GLOBAL_RNG : rng
    )

    dl_val = Flux.DataLoader(
        (processed_data.val.tensor, processed_data.val.labels),
        batchsize = batch_size,
        shuffle = false,
        partial = false,
        rng = isnothing(rng) ? Random.GLOBAL_RNG : MersenneTwister(rng)
    )
    
    dl_test = Flux.DataLoader(
        (processed_data.test.tensor, processed_data.test.labels),
        batchsize = batch_size,
        shuffle = false,
        partial = false,
        rng = isnothing(rng) ? Random.GLOBAL_RNG : MersenneTwister(rng)
    )
    
    return dl_train, dl_val, dl_test
end