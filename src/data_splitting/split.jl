"""
    leading_colons(x)

Return a tuple of `:` (colons) of length `ndims(x) - 1`.

Useful for slicing all but the last dimension of an array.

```julia
A = rand(3, 4, 5)
A[leading_colons(A)..., 2]  # selects all elements in the last dimension at index 2
```
"""
leading_colons(x) = ntuple(_ -> :, ndims(x) - 1)

"""
    train_val_test_split(data, labels; train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, shuffle=true, seed=nothing)
# Example
```julia
A = rand(3, 4, 5)
A[leading_colons(A)..., 2]  # selects all elements in the last dimension at index 2
```
"""
leading_colons(x) = ntuple(_ -> :, ndims(x) - 1)

"""
    train_val_test_split(data, labels; train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, shuffle=true, seed=nothing)

Split data and labels together into train/validation/test sets.

Uses `view()` for memory efficiency - returns lightweight views instead of copying data.

# Arguments  
- `raw_data`: Data to split (vectors, arrays, etc.)
- `labels`: Corresponding labels (vector or matrix where first dimension = number of samples)
- Other arguments same as index-based version

# Returns
- Named tuple with `(train=(data=..., labels=...), val=(...), test=(...))`
- All data and labels are views (SubArrays) for memory efficiency

# Examples
```julia
# Vector labels
sequences = ["ATCG", "GCTA", "TTAG"]  
labels = [0.1, 0.5, 0.9]
splits = train_val_test_split(sequences, labels; seed=42)

# Matrix labels (multi-target)
labels_matrix = [0.1 0.2; 0.5 0.6; 0.9 0.8]  # 3 samples, 2 targets each
splits = train_val_test_split(sequences, labels_matrix; seed=42)
train_labels = splits.train.labels  # Will be 2D matrix view

# Views are memory efficient but behave like regular arrays
println(typeof(splits.train.data))  # SubArray{...}
println(splits.train.data[1])       # Access works normally
```
"""
function train_val_test_split(
    data; 
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

    # Extract data and labels
    n_data_samples = size(data.X, ndims(data.X))
    n_label_samples = size(data.Y, ndims(data.Y))

    if n_data_samples != n_label_samples
        throw(ArgumentError("Data and labels must have same number of samples: $n_data_samples vs $n_label_samples"))
    end
    
    # Get indices
    indices = get_split_indices(n_data_samples; 
        train_ratio, val_ratio, test_ratio,
        _shuffle=_shuffle, 
        rng=rng)

    return (
        train = (X = view(data.X, leading_colons(data.X)..., indices.train), Y = view(data.Y, leading_colons(data.Y)..., indices.train)),
        val   = (X = view(data.X, leading_colons(data.X)..., indices.val),   Y = view(data.Y, leading_colons(data.Y)..., indices.val)),
        test  = (X = view(data.X, leading_colons(data.X)..., indices.test),  Y = view(data.Y, leading_colons(data.Y)..., indices.test))
    )

end