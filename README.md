# AutoComputationalGraphTuning

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kchu25.github.io/AutoComputationalGraphTuning.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kchu25.github.io/AutoComputationalGraphTuning.jl/dev/)
[![Build Status](https://github.com/kchu25/AutoComputationalGraphTuning.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kchu25/AutoComputationalGraphTuning.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kchu25/AutoComputationalGraphTuning.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kchu25/AutoComputationalGraphTuning.jl)

This package just requires two things: a `data` struct and a function to create a Flux model. 
Following duck typing practices, these must satisfy the following:

## Requirements

### Data Structure
Your `data` struct needs four fields:
- `data.X` - the features
- `data.Y` - the labels  
- `data.X_dim` - dimension of each feature
- `data.Y_dim` - dimension of each label

### Model Creation Function
You need a `create_model` function that:
- Takes `(X_dim, Y_dim, batch_size; rng, use_cuda)` as arguments
- Returns a Flux model
- Must define a `linear_sum` property

## Loss Function Configuration

Want to use a different loss function? No problem! You can configure any Flux loss function with custom aggregation:

```julia
# Default: MSE with mean aggregation
tune_hyperparameters(data, create_model)

# Use MAE instead
tune_hyperparameters(data, create_model; 
                    loss_fcn=(loss=Flux.mae, agg=StatsBase.mean))

# Huber loss (robust to outliers)
tune_hyperparameters(data, create_model;
                    loss_fcn=(loss=Flux.huber_loss, agg=StatsBase.mean))

# MSE with sum aggregation instead of mean
tune_hyperparameters(data, create_model;
                    loss_fcn=(loss=Flux.mse, agg=sum))
```

The `loss_fcn` parameter is a named tuple with:
- `loss`: Any Flux loss function (`Flux.mse`, `Flux.mae`, `Flux.huber_loss`, etc.)
- `agg`: Aggregation function (`StatsBase.mean`, `sum`, `identity`, or your own function)

This works for both `tune_hyperparameters()` and `train_final_model()` - just pass the same `loss_fcn` parameter!

## Basic Usage

```julia
# Tune hyperparameters - now returns the best model!
results_df, best_model, best_info = tune_hyperparameters(
    data, create_model; 
    max_epochs=50, 
    n_trials=100,
    save_folder="results/my_tuning_run"
)

# The best model is ready to use immediately
println("Best model achieved R² = $(best_info.r2)")
println("Best seed was: $(best_info.seed)")
println("Best batch size: $(best_info.batch_size)")

# Or train a final model with more epochs using best seed
model, stats = train_final_model(data, create_model; 
                                seed=best_info.seed, 
                                max_epochs=200)
```

## Advanced: Load Best Trial Configuration

When you run tuning with `save_folder`, the package automatically saves:
- CSV file with all trial results
- JSON file for each trial in a `json/` subfolder with complete configuration

You can load the best trial's configuration and use it for final training:

```julia
# Run tuning and save configs
tune_hyperparameters(data, create_model; 
                    n_trials=100,
                    save_folder="results/experiment_001")

# Later, load the best trial configuration
config = load_best_trial_config("results/experiment_001")

# Train final model using the exact same settings as the best trial
model, stats = train_final_model_from_config(data, create_model, config;
                                            max_epochs=200,
                                            patience=20)
```

Each trial's JSON config includes:
- Seed (for reproducibility)
- Normalization settings (`normalize_Y`, `normalization_method`, `normalization_mode`)
- Hardware settings (`use_cuda`)
- Batch size settings (`randomize_batchsize`, actual `batch_size` used)
- Loss function configuration (`loss_function`, `aggregation`)
- Trial results (`best_r2`, `val_loss`)

That's it! The package handles data splitting, normalization, early stopping, and gives you flexibility over loss functions while keeping the API simple.

