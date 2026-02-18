# Internal helpers for final model and code processor training

"""Prepare setup for final model training. Returns: (setup, batch_size)"""
function _prepare_final_model_setup(raw_data, create_model::Function;
                                    seed=1, randomize_batchsize=true,
                                    normalize_Y=true, normalization_method=:zscore,
                                    normalization_mode=:rowwise, use_cuda=true,
                                    loss_fcn=(loss=Flux.mse, agg=StatsBase.mean),
                                    model_kwargs...)
    rng = set_reproducible_seeds!(seed)
    batch_size = randomize_batchsize ? rand(rng, BATCH_SIZE_RANGE) : DEFAULT_BATCH_SIZE
    
    setup = setup_training(raw_data, create_model, batch_size; combine_train_val=true,
                          normalize_Y, normalization_method, normalization_mode, rng, 
                          use_cuda, loss_fcn, model_kwargs...)
    isnothing(setup) && error("Invalid hyperparameters for final model training")
    
    return setup, batch_size
end

"""Create dataloaders for final model training. Returns: (dl_train, dl_test)"""
function _create_final_dataloaders(setup, batch_size, seed)
    dl_train = Flux.DataLoader((setup.processed_data.train.tensor, setup.processed_data.train.labels),
                               batchsize=batch_size, shuffle=true, partial=false, rng=MersenneTwister(seed))
    dl_test = Flux.DataLoader((setup.processed_data.test.tensor, setup.processed_data.test.labels),
                              batchsize=batch_size, shuffle=false, partial=true)
    return dl_train, dl_test
end

"""
Create eval dataloaders with no shuffling and partial=true so every sample is included
and order matches split_indices. Returns: (dl_train_eval, dl_test_eval)
"""
function _create_eval_dataloaders(setup, batch_size)
    dl_train_eval = Flux.DataLoader((setup.processed_data.train.tensor, setup.processed_data.train.labels),
                                    batchsize=batch_size, shuffle=false, partial=true)
    dl_test_eval  = Flux.DataLoader((setup.processed_data.test.tensor, setup.processed_data.test.labels),
                                    batchsize=batch_size, shuffle=false, partial=true)
    return dl_train_eval, dl_test_eval
end

"""Train final model and load best weights. Returns: (trained_model, stats)"""
function _train_final_model!(setup, dl_train, dl_test; max_epochs=50, patience=10, print_every=100)
    println("🎯 Training final model (train+val combined)...")
    best_state, stats = train_model(setup.model, setup.opt_state, dl_train, dl_test, setup.Ydim;
                                    max_epochs, patience, print_every, test_set=true, loss_fcn=setup.loss_fcn)
    
    Flux.loadmodel!(setup.model_clone, best_state)
    return setup.model_clone, stats
end

"""Initialize processor and optimizer with seeded RNG"""
function _init_processor(proc_wrap, model_hp, use_hard_mask::Bool, seed::Int)
    rng = MersenneTwister(seed)
    processor = proc_wrap.create_processor(model_hp; arch_type=proc_wrap.arch_type, use_hard_mask=use_hard_mask, rng=rng)
    opt_state = Flux.setup(Flux.AdaBelief(), processor)
    return processor, opt_state
end

"""Compute predictions and gradients w.r.t. code"""
function _compute_code_gradients(model, code, proc_wrap, inf_layer::Int, predict_position::Int)
    (_, preds), grad = Flux.withgradient(code) do x
        linear_sum_fcn = @ignore model.predict_up_to_final_nonlinearity;
        preds = linear_sum_fcn(x; predict_position=predict_position)
        # preds = proc_wrap.predict_from_code(model, x; 
        #     layer=inf_layer, apply_nonlinearity=false, predict_position=predict_position)
        preds |> sum, preds
    end
    return preds, grad[1]
end

"""Single training step for code processor"""
function _processor_train_step!(processor, opt_state, model, seq, proc_wrap, inf_layer::Int, 
                                predict_position::Int, step::Int)
    code = model.code(seq |> gpu)
    preds, grad = _compute_code_gradients(model, code, proc_wrap, inf_layer, predict_position)
    
    loss, grads = Flux.withgradient(processor) do p
        proc_grad = p(code, grad; step=step)
        sum(abs2, vec(sum(proc_grad .* code, dims=(1,2))) - preds)
    end
    
    Flux.update!(opt_state, processor, grads[1])
    return loss
end

"""Train epoch for code processor"""
function _train_processor_epoch!(processor, opt_state, model, dataloader, proc_wrap, 
                                 inf_layer::Int, predict_position::Int, step::Ref{Int})
    epoch_loss = DEFAULT_FLOAT_TYPE(0)
    
    for (seq, _) in dataloader
        loss = _processor_train_step!(processor, opt_state, model, seq, proc_wrap, 
                                     inf_layer, predict_position, step[])
        epoch_loss += loss
        step[] += 1
    end
    
    return epoch_loss / length(dataloader)
end
