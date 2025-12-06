"""
    get_split_indices(data_size::Int; train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, shuffle=true, seed=nothing)

Return shuffled (or ordered) indices for train/validation/test splits.

# Arguments
- `data_size`: Number of data points
- `train_ratio`, `val_ratio`, `test_ratio`: Proportions for each split (must sum to 1)
- `shuffle`: Shuffle indices (default: true)
- `seed`: Random seed (optional)

# Returns
- Named tuple: `(train, val, test)` index vectors

# Throws
- `ArgumentError` if ratios are invalid or data_size <= 0

# Example
train_idx, val_idx, test_idx = get_split_indices(1000; seed=42)


"""
function get_split_indices(data_size::Int; 
                            train_ratio::Float64=0.8, 
                            val_ratio::Float64=0.1, 
                            test_ratio::Union{Float64,Nothing}=nothing,
                            _shuffle::Bool=true,
                            rng=Random.GLOBAL_RNG
                            )

    # Auto-calculate test_ratio if not provided
    if test_ratio === nothing
        test_ratio = 1.0 - train_ratio - val_ratio
        if test_ratio < 0
            throw(ArgumentError("train_ratio + val_ratio cannot exceed 1.0"))
        end
    end
    
    # Validation checks
    if data_size <= 0
        throw(ArgumentError("data_size must be positive, got $data_size"))
    end
    
    if any(ratio < 0 for ratio in [train_ratio, val_ratio, test_ratio])
        throw(ArgumentError("All ratios must be non-negative"))
    end
    
    total_ratio = train_ratio + val_ratio + test_ratio
    if !isapprox(total_ratio, 1.0, atol=1e-6)
        throw(ArgumentError("Ratios must sum to 1.0, got $total_ratio"))
    end
        
    # Create indices
    indices = _shuffle ? Random.shuffle(rng, 1:data_size) : collect(1:data_size)
    
    # Calculate split points (ensure we use all data points)
    train_end = round(Int, data_size * train_ratio)
    val_end = train_end + round(Int, data_size * val_ratio)
    
    # Ensure we don't exceed data_size due to rounding
    train_end = min(train_end, data_size)
    val_end = min(val_end, data_size)
    
    # Create splits
    train_indices = @view indices[1:train_end]
    val_indices = @view indices[train_end+1:val_end]
    test_indices = @view indices[val_end+1:end]
    
    return (train=train_indices, val=val_indices, test=test_indices)
end

