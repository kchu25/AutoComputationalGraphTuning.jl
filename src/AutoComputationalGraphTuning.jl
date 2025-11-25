module AutoComputationalGraphTuning


using Random
using DataFrames, CSV, Dates
using Flux, CUDA
using RealLabelNormalization
using StatsBase, Statistics
using JSON3, StructTypes
using Zygote
using Zygote: @ignore
# using ChainRulesCore: @ignore_derivatives


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
include("training/customized_losses.jl")
include("training/eval.jl")
include("training/loss.jl")
include("training/train.jl")

include("config_management.jl")
include("utils.jl")
include("setup.jl")
include("tuning.jl")
include("train_finalmodel.jl")
include("finetune.jl")
include("train_code_processor.jl")

export setup_model_and_training
export TrainingConfig, save_trial_config, load_trial_config, load_best_trial_config, config_to_loss_fcn
export train_final_model, train_final_model_from_config
export tune_hyperparameters
export finetune_model, finetune_model_from_config
export finetune_grad_loss  # Custom loss for gradient-based fine-tuning
export train_code_processor  # Train code processor for gradient transformations

end
