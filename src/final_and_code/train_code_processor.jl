"""
Train a code processor to learn gradient transformations.

# Arguments
- `model`: Pre-trained model (frozen during training)
- `dataloader`: Training DataLoader
- `proc_wrap`: Named tuple with (create_processor, arch_type, predict_from_code, process_code)
- `seed`: Random seed (default: 42)
- `max_epochs`: Training epochs (default: 20)
- `predict_position`: Position for prediction (default: 1)
- `use_hard_mask`: Use hard masking in processor (default: true)
- `inference_code_layer`: Layer for code inference (default: from model.hp)

# Returns
- `processor`: Trained code processor
- `loss_history`: Training loss per epoch

# Example
```julia
proc_wrap = (create_processor=create_code_processor, arch_type=:mbconv,
             predict_from_code=predict_from_code, process_code=process_code)
model, _, _, dl_train, _ = train_final_model(data, create_model; seed=42)
processor, losses = train_code_processor(model, dl_train, proc_wrap; max_epochs=20)
```
"""
function train_code_processor(model, dataloader, proc_wrap;
                             seed=42, max_epochs=20, predict_position=1, use_hard_mask=true,
                             inference_code_layer=nothing)
    
    # Setup
    set_reproducible_seeds!(seed)
    inf_layer = isnothing(inference_code_layer) ? model.hp.inference_code_layer : inference_code_layer
    processor, opt_state = _init_processor(proc_wrap, model.hp, use_hard_mask, seed)
    
    # Print header
    println("=" ^ 60)
    println("🔧 Training code processor (arch: $(proc_wrap.arch_type), use_hard_mask: $use_hard_mask)")
    println("Seed: $seed, Epochs: $max_epochs")
    println("=" ^ 60)
    
    # Training loop
    step = Ref(0)
    loss_history = DEFAULT_FLOAT_TYPE[]
    
    for epoch in 1:max_epochs
        avg_loss = _train_processor_epoch!(processor, opt_state, model, dataloader, proc_wrap,
                                          inf_layer, predict_position, step)
        push!(loss_history, avg_loss)
        println("Epoch $epoch: Loss = $(round(avg_loss, digits=6))")
    end
    
    # Print summary
    println("=" ^ 60)
    println("Code processor training complete!")
    println("Final loss: $(round(loss_history[end], digits=6))")
    println("=" ^ 60)
    
    return processor, loss_history
end
