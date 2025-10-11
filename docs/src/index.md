```@meta
CurrentModule = AutoComputationalGraphTuning
```

# AutoComputationalGraphTuning

Documentation for [AutoComputationalGraphTuning](https://github.com/kchu25/AutoComputationalGraphTuning.jl).

## Quick Start

This package makes hyperparameter tuning for Flux models simple. You just need a data struct and a model creation function.

### What You Need

**Data struct** with four fields:
- `data.X`, `data.Y` - your features and labels
- `data.X_dim`, `data.Y_dim` - dimensions of each feature/label

**Model function** that creates a Flux model:
```julia
function create_model(X_dim, Y_dim, batch_size; rng, use_cuda)
    # Your model code here
    return model  # Must have a `linear_sum` property
end
```

### Loss Function Flexibility

Want to experiment with different loss functions? Easy:

```julia
# Default MSE
tune_hyperparameters(data, create_model)

# Try MAE instead (less sensitive to outliers)
tune_hyperparameters(data, create_model; 
                    loss_fcn=(loss=Flux.mae, agg=StatsBase.mean))

# Huber loss (robust choice)
tune_hyperparameters(data, create_model;
                    loss_fcn=(loss=Flux.huber_loss, agg=StatsBase.mean))
```

The `loss_fcn` parameter takes any Flux loss function with your choice of aggregation. Works the same way in `train_final_model()` too.

### Basic Workflow

```julia
# 1. Tune hyperparameters
results = tune_hyperparameters(data, create_model; 
                              max_epochs=50, n_trials=100)

# 2. Train final model  
model, stats = train_final_model(data, create_model; 
                                seed=42, max_epochs=100)
```

The package handles data splitting, normalization, early stopping, and model selection automatically.

```@index
```

```@autodocs
Modules = [AutoComputationalGraphTuning]
```
