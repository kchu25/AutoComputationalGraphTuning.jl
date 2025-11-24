"""
Fine-tune a pre-trained model with custom loss function.

# Arguments
- `model`: Pre-trained model to fine-tune
- `raw_data`: Training data
- `compute_loss`: Custom loss function with signature (model, seq, labels, mask) -> (loss, aux_info)
- `seed`: Random seed for reproducibility
- `batch_size`: Batch size (if not provided, uses DEFAULT_BATCH_SIZE)
- `max_epochs`: Maximum training epochs
- `patience`: Early stopping patience
- `print_every`: Print frequency
- `learning_rate`: Learning rate for fine-tuning (typically smaller than initial training, default: 1e-4)
- `normalize_Y`: Whether to normalize Y
- `normalization_method`: Normalization method (:zscore or :minmax)
- `normalization_mode`: Normalization mode (:rowwise or :columnwise)
- `use_cuda`: Use GPU if available
- `combine_train_val`: Whether to combine train and validation sets for fine-tuning

# Returns
- `model`: Fine-tuned model (weights updated in place)
- `stats`: Training statistics
- `train_stats`: Normalization statistics

# Example
```julia
using AutoComputationalGraphTuning

# Load pre-trained model
model = ... # your pre-trained model

# Define custom loss
custom_loss = (m, x, y, mask) -> begin
    output = m(x)
    preds = output isa Tuple ? output[1] : output
    loss = masked_mse(preds, y, mask)
    (loss, Dict(:valid_count => sum(mask)))
end

# Fine-tune
finetuned_model, stats, train_stats = finetune_model(
    model, data;
    compute_loss = custom_loss,
    seed = 42,
    max_epochs = 20,
    learning_rate = 1e-4
)
```

# Using the gradient regularization loss
```julia
# Import custom loss from training module
using AutoComputationalGraphTuning: finetune_grad_loss

# Fine-tune with gradient penalty
finetuned_model, stats = finetune_model(
    model, data;
    compute_loss = (m, x, y, mask) -> finetune_grad_loss(
        m, x, y, mask; 
        predict_position=1, 
        grad_penalty_weight=0.1
    ),
    seed = 42,
    learning_rate = 5e-5  # Lower learning rate for gradient penalty
)
```
"""
function finetune_model(model, raw_data;
                       compute_loss=nothing,
                       seed=42,
                       batch_size=nothing,
                       max_epochs=20,
                       patience=5,
                       print_every=50,
                       learning_rate=1e-5,  # Lower default LR to prevent explosion
                       normalize_Y=true,
                       normalization_method=:zscore,
                       normalization_mode=:rowwise,
                       use_cuda=true,
                       combine_train_val=false)
    
    # Set random seed
    rng = set_reproducible_seeds!(seed)
    
    # Determine batch size
    batch_size = isnothing(batch_size) ? DEFAULT_BATCH_SIZE : batch_size
    
    println("=" ^ 60)
    println("Fine-tuning model with custom loss")
    println("Seed: $seed, Batch size: $batch_size")
    println("Learning rate: $learning_rate")
    println("Max epochs: $max_epochs, Patience: $patience")
    println("Print every: $print_every batches")
    println("Combine train+val: $combine_train_val")
    println("=" ^ 60)
    
    # Setup data processing only (no model creation)
    setup = setup_training(raw_data, nothing, batch_size;
                          normalize_Y, normalization_method, normalization_mode,
                          rng, use_cuda, combine_train_val,
                          loss_fcn=(loss=Flux.mse, agg=StatsBase.mean))
    
    if isnothing(setup)
        error("Failed to setup training data")
    end
    
    # Create optimizer with specified learning rate for the provided model
    opt = Flux.Adam(learning_rate)
    opt_state = Flux.setup(opt, model)
    
    # Get data loaders
    if combine_train_val
        dl_train, _, dl_test = obtain_data_loaders(setup.processed_data, batch_size; rng)
        dl_val = dl_test  # Use test set for validation when combining train+val
    else
        dl_train, dl_val, _ = obtain_data_loaders(setup.processed_data, batch_size; rng)
    end
    
    # Train with custom loss
    model_state, stats = train_model(
        model, opt_state, dl_train, dl_val, setup.Ydim;
        max_epochs, patience, print_every,
        compute_loss=compute_loss
    )
    
    # Load best model weights
    Flux.loadmodel!(model, model_state)
    
    println("\n" * "=" ^ 60)
    println("Fine-tuning complete!")
    println("Best validation R²: $(round(stats[:best_r2], digits=4))")
    println("Best validation loss: $(round(stats[:best_val_loss], digits=6))")
    println("Epochs trained: $(stats[:epochs_trained])")
    println("Converged: $(stats[:converged])")
    println("=" ^ 60)
    
    return model, stats, setup.train_stats
end


"""
Fine-tune a pre-trained model from a saved TrainingConfig.

# Arguments
- `model`: Pre-trained model to fine-tune
- `raw_data`: Training data
- `config`: TrainingConfig from previous training
- `compute_loss`: Custom loss function
- `max_epochs`: Maximum epochs for fine-tuning
- `patience`: Early stopping patience
- `print_every`: Print frequency
- `learning_rate`: Learning rate (default: 1e-4, typically lower than initial training)

# Returns
- `model`: Fine-tuned model
- `stats`: Training statistics

# Example
```julia
# Load config from previous training
config = load_training_config("path/to/config.json")

# Load pre-trained model
model = ... # your model

# Fine-tune with config settings but custom loss
finetuned_model, stats = finetune_model_from_config(
    model, data, config;
    compute_loss = my_custom_loss,
    max_epochs = 10,
    learning_rate = 5e-5
)
```
"""
function finetune_model_from_config(model, raw_data, config::TrainingConfig;
                                   compute_loss=nothing,
                                   max_epochs=20,
                                   patience=5,
                                   print_every=100,
                                   learning_rate=1e-4)
    
    finetune_model(model, raw_data;
                  compute_loss=compute_loss,
                  seed=config.seed,
                  batch_size=config.batch_size,
                  max_epochs=max_epochs,
                  patience=patience,
                  print_every=print_every,
                  learning_rate=learning_rate,
                  normalize_Y=config.normalize_Y,
                  normalization_method=config.normalization_method,
                  normalization_mode=config.normalization_mode,
                  use_cuda=config.use_cuda,
                  combine_train_val=false)  # Fine-tuning typically uses separate val set
end
