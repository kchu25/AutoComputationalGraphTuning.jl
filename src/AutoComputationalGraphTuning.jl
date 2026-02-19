module AutoComputationalGraphTuning


using Random
using DataFrames, CSV, Dates
using cuDNN 
using CUDA
using Flux
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
include("_data_splitting/structs.jl")
include("_data_splitting/indexing.jl")
include("_data_splitting/split.jl")

include("_training/log.jl")
include("_training/check_state.jl")
include("_training/customized_losses.jl")
include("_training/eval.jl")
include("_training/loss.jl")
include("_training/train.jl")

include("config_management.jl")
include("utils.jl")
include("setup.jl")
include("tune/tuning.jl")
include("final_and_code/train_finalmodel.jl")
include("final_and_code/train_code_processor.jl")
include("final_and_code/code_processor_eval.jl")
include("final_and_code/gyro_thresh.jl")

export setup_model_and_training
export TrainingConfig, save_trial_config, load_trial_config, load_best_trial_config
export compile_loss, create_masked_loss_function  # create_masked_loss_function is a backward compat alias
export train_final_model, train_final_model_from_config
export tune_hyperparameters
export finetune_model, finetune_model_from_config
export finetune_grad_loss  # Custom loss for gradient-based fine-tuning
export train_code_processor  # Train code processor for gradient transformations
export evaluate_code_processor  # Evaluate code processor performance

end
