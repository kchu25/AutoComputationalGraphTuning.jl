"""Statistics for processor evaluation"""
struct ProcessorEvalStats{T<:AbstractFloat}
    r2_original::T
    r2_processor::T
end

function Base.show(io::IO, stats::ProcessorEvalStats)
    println(io, "ProcessorEvalStats:")
    println(io, "  R² Original (gyro·code): ", round(stats.r2_original, digits=4))
    print(io, "  R² Processor:            ", round(stats.r2_processor, digits=4))
end

"""Compute gyros and predictions for a single batch"""
function _compute_gyro_and_preds(model, code, predict_position::Int)
    (_, preds), gyro = Flux.withgradient(code) do x
        linear_sum_fcn = @ignore model.predict_up_to_final_nonlinearity;
        preds = linear_sum_fcn(x; predict_position=predict_position)
        preds |> sum, preds # need the first component to be the gradient, but don't need to return it
    end
    return preds, gyro[1]
end

"""Compute R² coefficient"""
function _compute_r2(y_true::AbstractVector{T}, y_pred::AbstractVector{T}) where T<:AbstractFloat
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    return T(1) - ss_res / ss_tot
end

"""
Evaluate code processor performance on a dataset.

# Arguments
- `model`: Trained model
- `processor`: Trained code processor
- `dataloader`: DataLoader for evaluation
- `set_name`: Name of the dataset (e.g., "Train", "Test")
- `predict_position`: Position for prediction (default: 1)

# Returns
`ProcessorEvalStats` containing R² scores for original gradients vs processor gradients

# Example
```julia
stats_train = evaluate_processor(m, processor, dl_train, "Train")
stats_test = evaluate_processor(m, processor, dl_test, "Test")
```
"""
function evaluate_processor(model, processor, dataloader, set_name::String;
                           predict_position::Int=1)
    
    T = DEFAULT_FLOAT_TYPE
    
    # Collect predictions and products
    preds_collection = Vector{T}[]
    gyro_prods_collection = Vector{T}[]
    proc_prods_collection = Vector{T}[]
    
    for (seq, _) in dataloader
        code = model.code(seq |> gpu)
        preds, gyro = _compute_gyro_and_preds(model, code, predict_position)

        processor.training[] = false
        proc_gyro = processor(code, gyro)
        
        # Compute gyro·code products
        gyro_code_product = gyro .* code
        proc_gyro_code_product = proc_gyro .* code
        
        gyro_prod = vec(sum(gyro_code_product, dims=(1,2)))
        proc_prod = vec(sum(proc_gyro_code_product, dims=(1,2)))
        
        # Collect
        push!(preds_collection, vec(cpu(preds)))
        push!(gyro_prods_collection, cpu(gyro_prod))
        push!(proc_prods_collection, cpu(proc_prod))
    end
    
    # Concatenate
    preds_all = vcat(preds_collection...)
    gyro_prods = vcat(gyro_prods_collection...)
    proc_prods = vcat(proc_prods_collection...)
    
    # Compute R² scores
    r2_orig = _compute_r2(preds_all, gyro_prods)
    r2_proc = _compute_r2(preds_all, proc_prods)
    
    # Print results
    println("\n=== $set_name Set ===")
    println("R² scores:")
    println("  Original (gyro·code): $(round(r2_orig, digits=4))")
    println("  Processor: $(round(r2_proc, digits=4))")
    
    return ProcessorEvalStats(r2_orig, r2_proc)
end
