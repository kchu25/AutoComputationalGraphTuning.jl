"""
Train a code processor to learn gradient transformations.

# Arguments
- `model`: Pre-trained model with code processor architecture
- `dataloader`: DataLoader with training data
- `proc_wrap`: Named tuple containing processor functions and config with fields:
    - `create_processor`: Function to create processor
    - `arch_type`: Architecture type (e.g., :plain, :mbconv)
    - `predict_from_code`: Function to predict from code
    - `process_code`: Function to process code with gradient
- `seed`: Random seed (default: 42)
- `max_epochs`: Training epochs (default: 15)
- `predict_position`: Position for prediction (default: 1)
- `use_hard_mask`: Whether to use hard masking in processor (default: true)
- `inference_code_layer`: Layer for code inference (default: from model.hp)

# Returns
- `processor`: Trained code processor
- `loss_history`: Training loss per epoch

# Example
```julia
# Define processor wrapper
proc_wrap = (
    create_processor = create_code_processor,
    arch_type = :mbconv,
    predict_from_code = predict_from_code,
    process_code = process_code
)

# Train final model first
model, stats, train_stats, dl_train, dl_test = train_final_model(data, create_model; seed=42)

# Then train processor
processor, losses = train_code_processor(
    model, dl_train, proc_wrap;
    seed=42,
    max_epochs=20,
    use_hard_mask=true
)
```
"""
function train_code_processor(model, dataloader, proc_wrap;
                             seed=42,
                             max_epochs=20,
                             predict_position=1,
                             use_hard_mask=true,
                             inference_code_layer=nothing)
    
    set_reproducible_seeds!(seed)
    
    # Get inference layer from model if not provided
    inf_layer = isnothing(inference_code_layer) ? model.hp.inference_code_layer : inference_code_layer
    
    # Create processor and optimizer
    processor = proc_wrap.create_processor(
            model.hp; arch_type=proc_wrap.arch_type, use_hard_mask=use_hard_mask)
    opt_state = Flux.setup(Flux.AdaBelief(), processor)
    
    loss_history = DEFAULT_FLOAT_TYPE[]
    
    println("=" ^ 60)
    println("🔧 Training code processor (arch: $(proc_wrap.arch_type), use_hard_mask: $use_hard_mask)")
    println("Seed: $seed, Epochs: $max_epochs")
    println("=" ^ 60)
    
    step = 0
    for epoch in 1:max_epochs
        epoch_loss = 0.0f0
        
        for (seq, _) in dataloader
            code = model.code(seq |> gpu)
            
            # Compute gradient of predictions w.r.t. code
            (_, preds), g = Flux.withgradient(code) do x
                preds = proc_wrap.predict_from_code(model, x; 
                    layer=inf_layer,
                    apply_nonlinearity=false,
                    predict_position=predict_position)
                preds |> sum, preds
            end
            
            # Train processor
            loss, grads = Flux.withgradient(processor) do p
                 pg = proc_wrap.process_code(p, code, g[1]; step=step)
                sum(abs2, vec(sum(pg .* code, dims=(1,2))) - preds)
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
    println("Code processor training complete!")
    println("Final loss: $(round(loss_history[end], digits=6))")
    println("=" ^ 60)
    
    return processor, loss_history
end
