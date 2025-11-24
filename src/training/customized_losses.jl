function finetune_grad_loss(model, seq, labels, nan_mask, 
        predict_up_to_final_nonlinearity; 
        predict_position=1, 
        grad_penalty_weight=DEFAULT_FLOAT_TYPE(1.0),
        )

    # Use GPU-compatible indexing - keep as array slices, not views
    labels = labels[nan_mask]

    code = model.code(seq)

    predict_upto_fn = predict_up_to_final_nonlinearity(model, 
            code; predict_position=predict_position
        )

    predictions = model.final_nonlinearity.(predict_upto_fn)
    pred_loss = masked_mse(predictions, labels, nan_mask)

    grad = Zygote.@ignore Zygote.gradient(code) do x
        sum(predict_up_to_final_nonlinearity(
            model, x; predict_position=predict_position
        ))
    end[1]

    grad_prod = reshape(sum(grad .* code, dims=(1,2)), size(labels))

    # Square loss: sum of squared differences
    grad_loss = grad_penalty_weight * mean(sum(abs2, grad_prod - predict_upto_fn))

    total_loss = pred_loss + grad_loss
    
    (total_loss, Dict(
        :pred_loss => pred_loss,
        :grad_penalty => grad_loss,
        :valid_count => sum(nan_mask)
    ))
end


# function finetune_grad_loss(model, seq, labels, nan_mask; 
#         predict_position=1, grad_penalty_weight=DEFAULT_FLOAT_TYPE(1.0))

#     # Use GPU-compatible indexing - keep as array slices, not views
#     nan_mask = nan_mask[predict_position:predict_position, :]
#     labels = labels[predict_position:predict_position, :]

#     code = model.code(seq)
#     predict_upto_fn, back_fcn = Zygote.pullback(code) do x
#         model.predict_up_to_final_nonlinearity(
#             x; predict_position=predict_position
#         )
#     end

#     predictions = model.final_nonlinearity.(predict_upto_fn)
#     pred_loss = masked_mse(predictions, labels, nan_mask)

#     # Create gradient signal on same device as predict_upto_fn
#     grad_signal = @ignore CUDA.ones(eltype(predict_upto_fn), size(predict_upto_fn))
#     ∂code = back_fcn(grad_signal)[1]  # Get gradient w.r.t. code

#     # Square loss: sum of squared differences
#     grad_loss_term = (∂code .* code) .- predict_upto_fn
#     grad_penalty = sum(abs2, grad_loss_term)  # Square each element and sum

#     total_loss = pred_loss + grad_penalty_weight * grad_penalty
    
#     (total_loss, Dict(
#         :pred_loss => pred_loss,
#         :grad_penalty => grad_penalty,
#         :valid_count => sum(nan_mask)
#     ))
# end
