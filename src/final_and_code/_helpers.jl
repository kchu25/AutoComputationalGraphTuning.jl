# Internal helpers for final model and code processor training

"""
Prepare setup for final model training.

# Returns: (setup, batch_size)
"""
function _prepare_final_model_setup(raw_data, create_model::Function;
                                    seed=1, randomize_batchsize=true,
                                    normalize_Y=true, normalization_method=:zscore,
                                    normalization_mode=:rowwise, use_cuda=true,
                                    loss_fcn=(loss=Flux.mse, agg=StatsBase.mean),
                                    model_kwargs...)
    rng = set_reproducible_seeds!(seed)
    batch_size = randomize_batchsize ? rand(rng, BATCH_SIZE_RANGE) : DEFAULT_BATCH_SIZE
    
    setup = setup_training(raw_data, create_model, batch_size; combine_train_val=true,
                          normalize_Y, normalization_method, normalization_mode, rng, use_cuda, loss_fcn, model_kwargs...)
    isnothing(setup) && error("Invalid hyperparameters for final model training")
    
    return setup, batch_size
end

"""
Create dataloaders for final model training (train+val combined) and testing.

# Returns: (dl_train, dl_test)
"""
function _create_final_dataloaders(setup, batch_size, seed)
    dl_train = Flux.DataLoader((setup.processed_data.train.tensor, setup.processed_data.train.labels),
                               batchsize=batch_size, shuffle=true, partial=false, rng=MersenneTwister(seed))
    dl_test = Flux.DataLoader((setup.processed_data.test.tensor, setup.processed_data.test.labels),
                              batchsize=batch_size, shuffle=false, partial=false)
    return dl_train, dl_test
end

"""
Train the final model and load best weights into model_clone.

# Returns: (trained_model, stats)
"""
function _train_final_model!(setup, dl_train, dl_test;
                            max_epochs=50, patience=10, print_every=100)
    println("🎯 Training final model (train+val combined)...")
    best_state, stats = train_model(setup.model, setup.opt_state, dl_train, dl_test, setup.Ydim;
                                    max_epochs, patience, print_every, test_set=true, loss_fcn=setup.loss_fcn)
    
    Flux.loadmodel!(setup.model_clone, best_state)
    return setup.model_clone, stats
end

"""
Train code processor using the trained final model and training data.

# Returns: (processor, loss_history)
"""
function _train_code_processor!(model, dl_train, create_code_processor::Function;
                               arch_type,
                               seed=42,
                               max_epochs=15,
                               predict_position=1,
                               use_hard_mask=true,
                               inference_code_layer=nothing,
                               process_code_fn::Function,
                               predict_from_code_fn::Function)
    
    set_reproducible_seeds!(seed)
    
    # Get inference layer from model if not provided
    inf_layer = isnothing(inference_code_layer) ? model.hp.inference_code_layer : inference_code_layer
    
    # Create processor and optimizer
    processor = create_code_processor(model.hp; arch_type=arch_type, use_hard_mask=use_hard_mask)
    opt_state = Flux.setup(Flux.AdaBelief(), processor)
    
    loss_history = DEFAULT_FLOAT_TYPE[]
    
    println("=" ^ 60)
    println("🔧 Training code processor (arch: $arch_type, use_hard_mask: $use_hard_mask)")
    println("Seed: $seed, Epochs: $max_epochs")
    println("=" ^ 60)
    
    step = 0
    for epoch in 1:max_epochs
        epoch_loss = 0.0f0
        
        for (seq, _) in dl_train
            code = model.code(seq |> gpu)
            
            # Compute gradient of predictions w.r.t. code
            (_, preds), grad = Flux.withgradient(code) do x
                preds = predict_from_code_fn(model, x; 
                    layer=inf_layer,
                    apply_nonlinearity=false,
                    predict_position=predict_position)
                preds |> sum, preds
            end
            
            # Train processor
            loss, grads = Flux.withgradient(processor) do p
                proc_grad = process_code_fn(p, code, grad[1]; step=step)
                sum(abs2, vec(sum(proc_grad .* code, dims=(1,2))) - preds)
            end
            
            Flux.update!(opt_state, processor, grads[1])
            epoch_loss += loss
            step += 1
        end
        
        avg_loss = epoch_loss / length(dl_train)
        push!(loss_history, avg_loss)
        println("Epoch $epoch: Loss = $(round(avg_loss, digits=6))")
    end
    
    println("=" ^ 60)
    println("Code processor training complete!")
    println("Final loss: $(round(loss_history[end], digits=6))")
    println("=" ^ 60)
    
    return processor, loss_history
end
