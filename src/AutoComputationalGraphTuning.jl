module AutoComputationalGraphTuning


const DEFAULT_BATCH_SIZE = 128  # Default batch size if not specified
const BATCH_SIZE_RANGE = 32:16:256  # Possible batch sizes for random selection


# Subroutine for splitting the data into train/val/test sets.
include("data_splitting/structs.jl")
include("data_splitting/indexing.jl")
include("data_splitting/split.jl")


include("training/log.jl")
include("training/check_state.jl")
include("training/train.jl")


export setup_model_and_training

end
