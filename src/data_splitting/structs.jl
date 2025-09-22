"""
    DataSplit

Represents a single data split with tensor and labels.

# Fields
- `tensor`: Encoded sequence tensor (4D array)
- `labels`: Labels (vector or matrix)
- `stats`: Normalization statistics (only present for training data when normalized)
"""
mutable struct DataSplit{T,L,S}
    tensor::T
    labels::L
    stats::Union{S,Nothing}

    function DataSplit(tensor::T, labels::L) where {T,L}
        new{T,L,Nothing}(tensor, labels, nothing)
    end
    function DataSplit(tensor::T, labels::L, stats::S) where {T,L,S}
        new{T,L,S}(tensor, labels, stats)
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