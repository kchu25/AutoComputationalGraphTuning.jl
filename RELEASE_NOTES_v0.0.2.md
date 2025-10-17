@JuliaRegistrator register

Release notes:

## Breaking changes

### Loss function configuration now requires named tuple format

**Before (v0.0.1):** Loss functions were passed as plain functions
```julia
tune_hyperparameters(data, create_model; loss_fcn=masked_mse)
train_final_model(data, create_model; loss_fcn=my_loss_function)
```

**After (v0.0.2):** Loss functions must be passed as a named tuple with `loss` and `agg` fields
```julia
tune_hyperparameters(data, create_model; 
                    loss_fcn=(loss=Flux.mse, agg=StatsBase.mean))

train_final_model(data, create_model; 
                 loss_fcn=(loss=Flux.mae, agg=sum))
```

**Why this change?** The new format provides:
- Flexibility to use any Flux loss function (mse, mae, huber_loss, etc.)
- Configurable aggregation (mean, sum, or custom functions)
- Better reproducibility tracking (loss config saved in tuning results)

**Migration guide:**
- Replace `loss_fcn=masked_mse` with `loss_fcn=(loss=Flux.mse, agg=StatsBase.mean)` (default)
- For custom loss functions, wrap them: `loss_fcn=(loss=my_custom_loss, agg=StatsBase.mean)`
- Default behavior unchanged if you don't specify `loss_fcn`

## New features

- **JSON configuration management**: Automatic saving of trial configurations during tuning
  - Each trial's settings saved to `json/trial_seed_N.json`
  - Load best trial config with `load_best_trial_config(save_folder)`
  - Train final model from config with `train_final_model_from_config(data, create_model, config)`

- **Loss function tracking**: Loss function and aggregation now recorded in tuning results CSV

- **CUDA determinism warnings**: Automatic detection and warnings when CUDA deterministic mode is not enabled

- **Improved reproducibility**: Better RNG management and comprehensive documentation for achieving bit-for-bit reproducible results

## Documentation

- Added `CUDA_DETERMINISM.md` - Complete guide for reproducible GPU training
- Added `RANDOMIZATION_ANALYSIS.md` - Detailed analysis of non-determinism sources
- Updated README with loss configuration examples and reproducibility guidance
