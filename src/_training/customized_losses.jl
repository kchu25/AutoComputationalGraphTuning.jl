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
