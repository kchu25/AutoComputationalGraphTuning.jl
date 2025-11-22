# Training System Documentation

## Overview

The training system provides a flexible, extensible framework for training neural networks with early stopping, validation metrics, and custom loss functions. The core function `train_model` supports both standard supervised learning and advanced use cases like gradient penalties, multi-task learning, and custom regularization.

## Quick Start

### Basic Usage (Standard Supervised Learning)

```julia
using AutoComputationalGraphTuning

# Train with default masked MSE loss
model_state, stats = train_model(
    model, 
    opt_state, 
    train_dl, 
    val_dl, 
    output_dim;
    max_epochs = 50,
    patience = 10,
    print_every = 100
)

# Load best model
Flux.loadmodel!(model, model_state)
```

### Complete Workflow

```julia
# 1. Setup: Prepare data and initialize model
setup = setup_training(
    raw_data, 
    create_model, 
    batch_size;
    normalize_Y = true,
    normalization_method = :zscore,
    use_cuda = true
)

# 2. Train: Run training loop with early stopping
model_state, stats = train_model(
    setup.model,
    setup.opt_state,
    setup.train_dl,
    setup.val_dl,
    setup.Ydim;
    max_epochs = 50,
    patience = 10
)

# 3. Load best model weights
Flux.loadmodel!(setup.model, model_state)

# 4. Inspect training statistics
println("Best validation R²: ", stats[:best_r2])
println("Epochs trained: ", stats[:epochs_trained])
println("Converged: ", stats[:converged])
```

## Core Components

### 1. `train_model` - Main Training Function

**Purpose**: Train a model with early stopping and validation monitoring.

**Arguments**:
- `model`: Flux model to train
- `opt_state`: Optimizer state (from `Flux.setup(optimizer, model)`)
- `train_dl`: Training data loader
- `val_dl`: Validation data loader  
- `output_dim`: Number of output dimensions

**Keyword Arguments**:
- `max_epochs=50`: Maximum training epochs
- `patience=10`: Early stopping patience (epochs without improvement)
- `min_delta=1e-4`: Minimum improvement threshold
- `print_every=100`: Print progress every N batches
- `compute_loss=nothing`: Custom loss function (see below)
- `loss_fcn=masked_mse`: Backward compatible loss function

**Returns**:
- `best_model_state`: State dict of best model (lowest validation loss)
- `training_stats`: Dict containing:
  - `:train_losses` - Training loss per epoch
  - `:val_losses` - Validation loss per epoch
  - `:val_r2_scores` - Validation R² per epoch
  - `:best_val_loss` - Best validation loss achieved
  - `:best_r2` - Best R² score achieved
  - `:epochs_trained` - Total epochs completed
  - `:converged` - Whether early stopping triggered

### 2. `train_epoch!` - Single Epoch Training

**Purpose**: Train model for one full epoch over the training data.

**Arguments**:
- `model`: Model to train
- `opt_state`: Optimizer state
- `dataloader`: Data loader for training batches
- `epoch`: Current epoch number
- `print_every`: Print frequency
- `compute_loss=nothing`: Optional custom loss function

**Returns**:
- `epoch_avg_loss`: Average loss across all batches
- `epoch_aux`: Array of auxiliary info dicts from each batch

### 3. `train_batch!` - Single Batch Update

**Purpose**: Compute gradients and update model parameters for one batch.

**Arguments**:
- `model`: Model to update
- `opt_state`: Optimizer state
- `seq, labels`: Batch data (sequences and labels)
- `compute_loss=nothing`: Optional custom loss function

**Returns**:
- `loss`: Scalar loss value for this batch
- `aux_info`: Dict with auxiliary information (e.g., `:valid_count`)

## Custom Loss Functions

### Signature

Custom loss functions must follow this signature:

```julia
function my_custom_loss(model, seq, labels, nan_mask)
    # Your loss computation here
    loss = ...
    aux_info = Dict(:valid_count => sum(nan_mask), ...)
    return (loss, aux_info)
end
```

**Parameters**:
- `model`: The neural network model
- `seq`: Input sequences (already on GPU if using CUDA)
- `labels`: Target labels (already on GPU if using CUDA)
- `nan_mask`: Boolean mask indicating valid (non-NaN) labels

**Returns**:
- `loss`: Scalar loss value (Float32)
- `aux_info`: Dict with auxiliary information for logging
  - Must include `:valid_count` for proper logging
  - Can include custom metrics (e.g., `:grad_penalty`, `:regularizer`)

### Example 1: Gradient Penalty Regularization

```julia
function gradient_penalty_loss(model, seq, labels, nan_mask; λ=0.1)
    # Forward pass
    preds = model(seq)
    
    # Standard prediction loss
    pred_loss = masked_mse(preds, labels, nan_mask)
    
    # Gradient penalty on predictions w.r.t. inputs
    grad_penalty = sum(abs2, Flux.gradient(x -> sum(model(x)), seq)[1])
    
    # Combined loss
    total_loss = pred_loss + λ * grad_penalty
    
    # Auxiliary info for logging
    aux_info = Dict(
        :valid_count => sum(nan_mask),
        :pred_loss => pred_loss,
        :grad_penalty => grad_penalty
    )
    
    return (total_loss, aux_info)
end

# Use it
train_model(model, opt_state, train_dl, val_dl, ydim;
    compute_loss = (m, x, y, mask) -> gradient_penalty_loss(m, x, y, mask; λ=0.05)
)
```

### Example 2: Multi-Output Model

```julia
function multi_output_loss(model, seq, labels, nan_mask)
    # Model returns (predictions, activations)
    preds, activations = model(seq)
    
    # Loss on predictions
    pred_loss = masked_mse(preds, labels, nan_mask)
    
    # Regularization on activations (e.g., sparsity)
    activation_penalty = 0.01 * sum(abs, activations)
    
    total_loss = pred_loss + activation_penalty
    
    aux_info = Dict(
        :valid_count => sum(nan_mask),
        :pred_loss => pred_loss,
        :activation_penalty => activation_penalty
    )
    
    return (total_loss, aux_info)
end
```

### Example 3: Attention Regularization

```julia
function attention_regularized_loss(model, seq, labels, nan_mask; λ_attn=0.01)
    # Model returns (predictions, attention_weights)
    preds, attn = model(seq; return_attention=true)
    
    # Prediction loss
    pred_loss = masked_mse(preds, labels, nan_mask)
    
    # Encourage sparse attention (L1 penalty)
    attn_sparsity = sum(abs, attn)
    
    total_loss = pred_loss + λ_attn * attn_sparsity
    
    aux_info = Dict(
        :valid_count => sum(nan_mask),
        :pred_loss => pred_loss,
        :attn_sparsity => attn_sparsity
    )
    
    return (total_loss, aux_info)
end
```

## Backward Compatibility

The system maintains backward compatibility with the older `loss_fcn` parameter:

```julia
# Old way (still works)
train_model(model, opt_state, train_dl, val_dl, ydim;
    loss_fcn = masked_mse
)

# New way (more flexible)
train_model(model, opt_state, train_dl, val_dl, ydim;
    compute_loss = my_custom_loss
)
```

Internally, `loss_fcn` is automatically converted to the new `compute_loss` format.

## Training Flow

1. **Initialization**
   - Set up early stopping variables
   - Initialize training history tracking

2. **Epoch Loop**
   - Call `train_epoch!` to train one full epoch
   - Compute average valid count from batch auxiliary info
   - Evaluate validation metrics (loss and R²)
   - Update training history
   - Print epoch summary
   - Check early stopping criteria

3. **Early Stopping**
   - Monitors validation loss
   - Saves best model state when validation improves
   - Stops if no improvement for `patience` epochs
   - Minimum improvement threshold controlled by `min_delta`

4. **Return Results**
   - Best model state (for loading with `Flux.loadmodel!`)
   - Training statistics dictionary

## Training Statistics

The returned `training_stats` dictionary contains:

```julia
Dict(
    :train_losses => [0.8, 0.6, 0.5, ...],        # Per-epoch training losses
    :val_losses => [0.85, 0.65, 0.52, ...],       # Per-epoch validation losses
    :val_r2_scores => [0.3, 0.5, 0.6, ...],       # Per-epoch validation R²
    :best_val_loss => 0.45,                        # Best validation loss
    :best_r2 => 0.68,                              # Best R² score
    :epochs_trained => 25,                         # Total epochs completed
    :converged => true                             # Whether early stopping triggered
)
```

## Common Patterns

### Loading Best Model

```julia
model_state, stats = train_model(...)
Flux.loadmodel!(model, model_state)  # Load best weights
```

### Monitoring Training Progress

```julia
# During training, progress is printed:
# Epoch 1, Batch 100: Loss = 0.823, Avg = 0.845, Valid: 48
# Epoch 1, Batch 200: Loss = 0.756, Avg = 0.789, Valid: 48
# ...
# Epoch 1 Summary:
#   Train Loss = 0.815318
#   Val Loss = 0.852341
#   Aggregated R² = 0.3245
#   Individual R² Mean = 0.3156 (45/48 feature(s))
#   Avg Valid Entries = 47.8
```

### Analyzing Results

```julia
model_state, stats = train_model(...)

# Check convergence
if stats[:converged]
    println("Training converged via early stopping")
else
    println("Training completed all epochs without convergence")
end

# Plot training curves
using Plots
plot(stats[:train_losses], label="Train Loss")
plot!(stats[:val_losses], label="Val Loss")

# Find best epoch
best_epoch = argmin(stats[:val_losses])
println("Best model at epoch: ", best_epoch)
```

## Advanced Use Cases

See `examples/custom_loss_examples.jl` for comprehensive examples including:

1. **Gradient Penalty Regularization** - Penalize gradients of outputs w.r.t. inputs
2. **Multi-Task Learning** - Train on multiple related tasks with weighted losses
3. **Attention Regularization** - Encourage sparse or specific attention patterns
4. **Contrastive Learning** - Add representation learning objectives
5. **Variational Methods** - KL divergence regularization (VAE-style)
6. **Curriculum Learning** - Progressively increase task difficulty
7. **Composite Losses** - Combine multiple loss components with weights

## Tips

1. **GPU Usage**: Data is automatically moved to GPU in `train_batch!` if CUDA is available
2. **NaN Handling**: The masked loss functions automatically handle NaN values in labels
3. **Memory**: Consider batch size if running out of GPU memory
4. **Reproducibility**: Use `set_reproducible_seeds!` before setup for deterministic results
5. **Custom Metrics**: Use `aux_info` dict to track any custom metrics during training
6. **Model Architecture**: Models can return tuples for multi-output scenarios
