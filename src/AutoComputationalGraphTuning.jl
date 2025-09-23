module AutoComputationalGraphTuning


using Random
using DataFrames, CSV, Dates
using Flux, CUDA
using RealLabelNormalization
using StatsBase

const DEFAULT_FLOAT_TYPE = Float32  # Default floating point type
const FLUX_MODEL_FLOAT_FCN = Flux.f32 # just to ensure that it matches DEFAULT_FLOAT_TYPE
const DEFAULT_BATCH_SIZE = 128  # Default batch size if not specified
const BATCH_SIZE_RANGE = 32:16:256  # Possible batch sizes for random selection


# Subroutine for splitting the data into train/val/test sets.
include("data_splitting/structs.jl")
include("data_splitting/indexing.jl")
include("data_splitting/split.jl")

include("training/log.jl")
include("training/check_state.jl")
include("training/eval.jl")
include("training/loss.jl")
include("training/train.jl")


include("utils.jl")
include("setup.jl")
include("tuning.jl")
include("train_finalmodel.jl")

export setup_model_and_training

end
