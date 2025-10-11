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
# Tune hyperparameters
results = tune_hyperparameters(data, create_model; 
                              max_epochs=50, 
                              n_trials=100)

# Train final model with best hyperparameters
model, stats = train_final_model(data, create_model; 
                                seed=42, 
                                max_epochs=100)
```

That's it! The package handles data splitting, normalization, early stopping, and gives you flexibility over loss functions while keeping the API simple.

