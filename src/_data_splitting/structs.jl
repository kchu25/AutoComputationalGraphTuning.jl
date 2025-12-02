# Helper function for type conversion
_to_default_float(x) = Array{DEFAULT_FLOAT_TYPE, ndims(x)}(x)

"""
    DataSplit

Represents a single data split with tensor and labels.

# Fields
- `tensor`: Encoded sequence tensor (e.g. 4D array)
- `labels`: Labels (vector or matrix)
- `stats`: Normalization statistics (only present for training data when normalized)
"""
mutable struct DataSplit{T,L,S}
    tensor::T
    labels::L
    stats::Union{S,Nothing}

    function DataSplit(tensor, labels, stats=nothing)
        converted_tensor = _to_default_float(tensor)
        converted_labels = _to_default_float(labels)
        new{typeof(converted_tensor), typeof(converted_labels), typeof(stats)}(
            converted_tensor,
            converted_labels,
            stats
        )
    end
end

"""
    PreprocessedData

Container for preprocessed train/validation/test splits.

# Fields  
- `train`: Training data split
- `val`: Validation data split
- `test`: Test data split
"""
struct PreprocessedData{T,V,S}
    train::T
    val::V
    test::S
end