# Breaking Changes

## Loss Function Naming (Clarity Update)

To reduce confusion about loss function types, parameter names have been clarified:

### Summary of Changes

| Old Name | New Name | What It Is | Location |
|---|---|---|---|
| `loss_fcn` (parameter) | **`loss_spec`** | Loss specification NamedTuple `(loss=Flux.mse, agg=StatsBase.mean)` | User-facing APIs |
| `create_masked_loss_function()` | **`compile_loss()`** | Function that compiles a spec into a callable | `src/_training/loss.jl` |
| `setup.loss_fcn` | **`setup.compiled_loss`** | The compiled 3-arg loss function | Internal setup object |
| `loss_fcn` (train_model param) | **`compiled_loss`** | Pre-compiled loss function `(preds, targets, mask) → scalar` | `train_model()` |
| `DEFAULT_LOSS_CONFIG` | **`DEFAULT_LOSS_SPEC`** | Default loss specification constant | `src/_training/loss.jl` |

### User Code Changes Required

#### 1. Tune Hyperparameters
**Before:**
```julia
results, best_model, best_info = tune_hyperparameters(
    data, create_model;
    loss_fcn=(loss=Flux.mse, agg=StatsBase.mean),
    max_epochs=50
)
```

**After:**
```julia
results, best_model, best_info = tune_hyperparameters(
    data, create_model;
    loss_spec=(loss=Flux.mse, agg=StatsBase.mean),  # ← renamed parameter
    max_epochs=50
)
```

#### 2. Train Final Model
**Before:**
```julia
model, stats, train_stats, dl_train, dl_test = train_final_model(
    data, create_model;
    loss_fcn=(loss=Flux.mae, agg=StatsBase.mean),
    seed=42
)
```

**After:**
```julia
model, stats, train_stats, dl_train, dl_test = train_final_model(
    data, create_model;
    loss_spec=(loss=Flux.mae, agg=StatsBase.mean),  # ← renamed parameter
    seed=42
)
```

#### 3. Train Model (Advanced Usage)
**Before:**
```julia
best_state, stats = train_model(
    model, opt_state, train_dl, val_dl, output_dim;
    loss_fcn=my_compiled_loss_function
)
```

**After:**
```julia
best_state, stats = train_model(
    model, opt_state, train_dl, val_dl, output_dim;
    compiled_loss=my_compiled_loss_function  # ← renamed parameter
)
```

#### 4. Custom Loss Compilation (Advanced)
**Before:**
```julia
loss_fn = create_masked_loss_function((loss=Flux.mse, agg=StatsBase.mean))
```

**After:**
```julia
loss_fn = compile_loss((loss=Flux.mse, agg=StatsBase.mean))
```

> **Note:** `create_masked_loss_function` remains as a backward-compatible alias, but `compile_loss` is the preferred name going forward.

### Removed Exports

The following function was removed from public exports (it was commented-out code):
- `config_to_loss_fcn()` — **removed** (use explicit loss specs instead)

### Backward Compatibility

- The alias `create_masked_loss_function = compile_loss` is maintained — old code using the old function name will still work.
- The default value `masked_mse` remains unchanged.
- Internal loss computation (`masked_loss`, `masked_mse`) is unchanged.

### Migration Guide

**Quick checklist:**
- [ ] Replace all `loss_fcn=` with `loss_spec=` in `tune_hyperparameters()` calls
- [ ] Replace all `loss_fcn=` with `loss_spec=` in `train_final_model()` calls
- [ ] Replace all `loss_fcn=` with `compiled_loss=` in `train_model()` calls
- [ ] Replace `create_masked_loss_function()` with `compile_loss()` (or keep using the alias)
- [ ] Replace references to `setup.loss_fcn` with `setup.compiled_loss` if using internal APIs

### Why This Change?

The naming was confusing because `loss_fcn` referred to **two different types**:
1. **At user entry point:** A NamedTuple config `(loss=Flux.mse, agg=StatsBase.mean)`
2. **After compilation:** A 3-argument callable `(preds, targets, mask) → scalar`

The new names clearly distinguish:
- **`loss_spec`** = the specification/config (what users pass in)
- **`compile_loss`** = the compilation function (spec → callable)
- **`compiled_loss`** = the result (ready-to-use function)

This prevents the confusion that arose from asking "wait, does `loss_fcn` take 2 or 3 arguments?"
