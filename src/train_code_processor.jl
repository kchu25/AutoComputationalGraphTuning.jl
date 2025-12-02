"""
Train a code processor to learn gradient transformations.

# Arguments
- `model`: Pre-trained model with code processor architecture
- `dataloader`: DataLoader with training data
- `model_module`: Module containing model functions (e.g., VeryBasicCNN2)
- `arch_type`: Architecture type for code processor (e.g., :plain, :mbconv)
- `seed`: Random seed
- `max_epochs`: Training epochs (default: 15)
- `learning_rate`: Learning rate (default: 1e-3, not used with AdaBelief default settings)
- `predict_position`: Position for prediction (default: 1)
- `use_hard_mask`: Whether to use hard masking in processor (default: true)
- `inference_code_layer`: Layer for code inference (default: from model.hp)

# Returns
- `processor`: Trained code processor
- `loss_history`: Training loss per epoch

# Example
```julia
processor, losses = train_code_processor(
    model, dataloader, VeryBasicCNN2;
    arch_type=:mbconv,
    use_hard_mask=true,
    seed=42,
    max_epochs=20
)
```
"""
function train_code_processor(model, dataloader, model_module::Module;
                             arch_type,  # Required - user must specify
                             seed=42,
                             max_epochs=15,
                             learning_rate=1e-3,
                             predict_position=1,
                             use_hard_mask=true,
                             inference_code_layer=nothing)
    
    set_reproducible_seeds!(seed)
    
    # Get inference layer from model if not provided
    inf_layer = isnothing(inference_code_layer) ? model.hp.inference_code_layer : inference_code_layer
    
    # Create processor and optimizer using module
    processor = model_module.create_code_processor(model.hp; arch_type=arch_type, use_hard_mask=use_hard_mask)
    opt_state = Flux.setup(Flux.AdaBelief(), processor)
    
    loss_history = DEFAULT_FLOAT_TYPE[]
    
    println("=" ^ 60)
    println("Training code processor (arch: $arch_type, use_hard_mask: $use_hard_mask)")
    println("Seed: $seed, Epochs: $max_epochs")
    println("=" ^ 60)
    
    step = 0
    for epoch in 1:max_epochs
        epoch_loss = 0.0f0
        
        for (seq, _) in dataloader
            code = model.code(seq |> gpu)
            
            # Compute gradient of predictions w.r.t. code
            preds, grad = Flux.withgradient(code) do x
                preds = model_module.predict_from_code(model, x; 
                    layer=inf_layer,
                    apply_nonlinearity=false,
                    predict_position=predict_position)
                preds
            end
            
            # Train processor
            loss, grads = Flux.withgradient(processor) do p
                c = model_module.process_code_with_gradient(p, code, grad[1]; step=step)
                sum(abs2, vec(sum(c .* code, dims=(1,2))) - preds)
            end
            
            Flux.update!(opt_state, processor, grads[1])
            epoch_loss += loss
            step += 1
        end
        
        avg_loss = epoch_loss / length(dataloader)
        push!(loss_history, avg_loss)
        println("Epoch $epoch: Loss = $(round(avg_loss, digits=6))")
    end
    
    println("=" ^ 60)
    println("Training complete!")
    println("Final loss: $(round(loss_history[end], digits=6))")
    println("=" ^ 60)
    
    return processor, loss_history
end
