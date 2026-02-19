# Configuration management for tuning and training

using JSON3
using StructTypes

"""
    get_function_name(f)

Get the fully qualified name of a function, preserving module prefix.
For common StatsBase re-exports (like mean), prefer StatsBase module name.
"""
function get_function_name(f)
    mod = parentmodule(f)
    fname = nameof(f)
    
    # Special handling for StatsBase re-exports
    # If it's mean/median/etc from Statistics but we use StatsBase, prefer StatsBase
    if mod == Statistics && fname in [:mean, :median, :std, :var]
        return "StatsBase.$(fname)"
    end
    
    return "$(mod).$(fname)"
end

"""
    TrainingConfig

Configuration for model training, including all hyperparameters and settings.
This struct is used to save/load trial configurations for reproducibility.
"""
Base.@kwdef struct TrainingConfig
    seed::Int
    normalize_Y::Bool = true
    normalization_method::Symbol = :zscore
    normalization_mode::Symbol = :rowwise
    use_cuda::Bool = true
    randomize_batchsize::Bool = true
    batch_size::Union{Int, Nothing} = nothing  # Actual batch size used (if not randomized)
    loss_function::String = "Flux.mse"
    aggregation::String = "StatsBase.mean"
    best_r2::Union{Float64, Nothing} = nothing
    val_loss::Union{Float64, Nothing} = nothing
end

# Enable JSON serialization
StructTypes.StructType(::Type{TrainingConfig}) = StructTypes.Struct()

"""
    save_trial_config(config::TrainingConfig, save_folder::String)

Save a trial configuration to a JSON file in the `json` subfolder.
"""
function save_trial_config(config::TrainingConfig, save_folder::String)
    json_folder = joinpath(save_folder, "json")
    mkpath(json_folder)
    
    json_file = joinpath(json_folder, "trial_seed_$(config.seed).json")
    
    try
        open(json_file, "w") do io
            JSON3.write(io, config)
        end
        return json_file
    catch e
        @warn "Could not save trial config to JSON" exception=e
        return nothing
    end
end

"""
    load_trial_config(json_path::String)

Load a trial configuration from a JSON file.
"""
function load_trial_config(json_path::String)
    try
        json_data = JSON3.read(read(json_path, String), TrainingConfig)
        return json_data
    catch e
        error("Could not load trial config from $json_path: $e")
    end
end

"""
    load_best_trial_config(save_folder::String)

Load the configuration for the best trial (highest R²) from a tuning run.
Reads the CSV results file and loads the corresponding JSON config.
"""
function load_best_trial_config(save_folder::String)
    # Find the most recent results CSV file
    csv_files = filter(x -> occursin("hyperparameter_results", x) && endswith(x, ".csv"), 
                      readdir(save_folder, join=true))
    
    if isempty(csv_files)
        error("No hyperparameter results CSV found in $save_folder")
    end
    
    # Get the most recent file
    csv_file = sort(csv_files, by=mtime, rev=true)[1]
    
    # Read results and find best trial
    results = CSV.read(csv_file, DataFrame)
    
    if nrow(results) == 0
        error("No trials found in results file")
    end
    
    # Sort by best_r2 and get the top seed
    sort!(results, :best_r2, rev=true)
    best_seed = results[1, :seed]
    
    # Load the corresponding JSON config
    json_file = joinpath(save_folder, "json", "trial_seed_$(best_seed).json")
    
    if !isfile(json_file)
        error("JSON config file not found for best trial (seed=$best_seed)")
    end
    
    return load_trial_config(json_file)
end

# """
#     config_to_loss_fcn(config::TrainingConfig)

# Convert a TrainingConfig's loss function strings back to a named tuple.
# Assumes the loss function and aggregation are from Flux/StatsBase.
# """
# function config_to_loss_fcn(config::TrainingConfig)
#     # Parse loss function
#     loss = eval(Meta.parse(config.loss_function))
#     agg = eval(Meta.parse(config.aggregation))
    
#     return (loss=loss, agg=agg)
# end
