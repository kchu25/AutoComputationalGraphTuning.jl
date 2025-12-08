"""Statistics for processor evaluation"""
struct ProcessorEvalStats{T<:AbstractFloat}
    r2_original::T
    r2_processor::T
    r2_original_nonsparse::T
    r2_processor_nonsparse::T
    gyro_nonzero_per_datapt::T
    proc_gyro_nonzero_per_datapt::T
    gyroprod_nonzero_per_datapt::T
    procprod_nonzero_per_datapt::T
    components_per_datapt::Int
    gyro_shape::Tuple{Int,Int,Int,Int}
end

"""Compute gyros and predictions for a single batch"""
function _compute_gyro_and_preds(model, code, proc_wrap, inf_layer::Int, predict_position::Int)
    (_, preds), gyro = Flux.withgradient(code) do x
        linear_sum_fcn = @ignore model.predict_up_to_final_nonlinearity;
        preds = linear_sum_fcn(x; predict_position=predict_position)
        # preds = proc_wrap.predict_from_code(model, x; 
        #     layer=inf_layer, apply_nonlinearity=false, predict_position=predict_position)
        preds |> sum, preds # need the first component to be the gradient, but don't need to return it
    end
    return preds, gyro[1]
end

"""Accumulate sparsity statistics from a batch"""
function _accumulate_batch_stats!(stats_dict, code, gyro, proc_gyro, preds, epsilon::AbstractFloat)
    # Count sparsity in gyros (before dot product)
    stats_dict[:gyro_sparse_count] += sum(abs.(gyro) .< epsilon)
    stats_dict[:proc_sparse_count] += sum(abs.(proc_gyro) .< epsilon)
    
    # Compute gyro·code products
    gyro_code_product = gyro .* code
    proc_gyro_code_product = proc_gyro .* code
    
    # Count sparsity in gyro·code products (before summing)
    stats_dict[:gyroprod_sparse_count] += sum(abs.(gyro_code_product) .< epsilon)
    stats_dict[:procprod_sparse_count] += sum(abs.(proc_gyro_code_product) .< epsilon)
    
    # Sum to get final predictions
    gyro_prod = vec(sum(gyro_code_product, dims=(1,2)))
    proc_prod = vec(sum(proc_gyro_code_product, dims=(1,2)))
    
    # Store masks for non-sparse components (using original gyro as reference)
    gyro_nonsparse_mask = abs.(gyro) .>= epsilon
    
    # Compute products only for non-sparse gyro components
    gyro_code_prod_nonsparse = gyro_code_product .* gyro_nonsparse_mask
    proc_gyro_code_prod_nonsparse = proc_gyro_code_product .* gyro_nonsparse_mask
    
    gyro_prod_nonsparse = vec(sum(gyro_code_prod_nonsparse, dims=(1,2)))
    proc_prod_nonsparse = vec(sum(proc_gyro_code_prod_nonsparse, dims=(1,2)))
    
    push!(stats_dict[:preds_collection], preds)
    push!(stats_dict[:gyro_prods], gyro_prod)
    push!(stats_dict[:proc_prods], proc_prod)
    push!(stats_dict[:gyro_prods_nonsparse], gyro_prod_nonsparse)
    push!(stats_dict[:proc_prods_nonsparse], proc_prod_nonsparse)
    stats_dict[:gyro_total_count] += length(gyro)
    stats_dict[:num_datapoints] += size(gyro, 4)
    
    return size(gyro)
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
- `proc_wrap`: Named tuple with (predict_from_code, process_code)
- `set_name`: Name of the dataset (e.g., "Train", "Test")
- `epsilon`: Threshold for sparsity (default: 1e-3)
- `inference_code_layer`: Layer for code inference (default: from model.hp)
- `predict_position`: Position for prediction (default: 1)

# Returns
`ProcessorEvalStats` containing R² scores and sparsity metrics
"""
function evaluate_processor(model, processor, dataloader, proc_wrap, set_name::String;
                           epsilon::DEFAULT_FLOAT_TYPE=5f-3,
                           inference_code_layer=nothing,
                           predict_position::Int=1)
    
    T = typeof(epsilon)
    inf_layer = isnothing(inference_code_layer) ? model.hp.inference_code_layer : inference_code_layer
    
    # Initialize statistics collectors
    stats = Dict(
        :preds_collection => Vector{T}[],
        :gyro_prods => Vector{T}[],
        :proc_prods => Vector{T}[],
        :gyro_prods_nonsparse => Vector{T}[],
        :proc_prods_nonsparse => Vector{T}[],
        :gyro_sparse_count => 0,
        :proc_sparse_count => 0,
        :gyroprod_sparse_count => 0,
        :procprod_sparse_count => 0,
        :gyro_total_count => 0,
        :num_datapoints => 0
    )
    
    gyro_shape = nothing
    
    for (seq, _) in dataloader
        code = model.code(seq |> gpu)
        preds, gyro = _compute_gyro_and_preds(model, code, proc_wrap, inf_layer, predict_position)
        
        if isnothing(gyro_shape)
            gyro_shape = size(gyro)
        end

        processor.training[] = false  # Set processor to eval mode

        proc_gyro = processor(code, gyro)
        
        _accumulate_batch_stats!(stats, code, gyro, proc_gyro, preds, epsilon)
    end
    
    # Aggregate predictions
    preds_all = cpu(vcat(stats[:preds_collection]...))
    gyro_prods = cpu(vcat(stats[:gyro_prods]...))
    proc_prods = cpu(vcat(stats[:proc_prods]...))
    gyro_prods_nonsparse = cpu(vcat(stats[:gyro_prods_nonsparse]...))
    proc_prods_nonsparse = cpu(vcat(stats[:proc_prods_nonsparse]...))
    
    # Compute R² scores (all components)
    r2_orig = _compute_r2(preds_all, gyro_prods)
    r2_proc = _compute_r2(preds_all, proc_prods)
    
    # Compute R² scores (non-sparse components only)
    r2_orig_nonsparse = _compute_r2(preds_all, gyro_prods_nonsparse)
    r2_proc_nonsparse = _compute_r2(preds_all, proc_prods_nonsparse)
    
    # Compute per-datapoint sparsity metrics
    components_per_datapt = prod(gyro_shape[1:3])
    gyro_nonzero_per_datapt = T((stats[:gyro_total_count] - stats[:gyro_sparse_count]) / stats[:num_datapoints])
    proc_gyro_nonzero_per_datapt = T((stats[:gyro_total_count] - stats[:proc_sparse_count]) / stats[:num_datapoints])
    
    gyroprod_total_count = stats[:gyro_total_count]  # same shape as gyro
    gyroprod_nonzero_per_datapt = T((gyroprod_total_count - stats[:gyroprod_sparse_count]) / stats[:num_datapoints])
    procprod_nonzero_per_datapt = T((gyroprod_total_count - stats[:procprod_sparse_count]) / stats[:num_datapoints])
    
    # Print results
    println("\n=== $set_name Set ===")
    println("Note: Statistics computed on $set_name set (independent from training)")
    println()
    println("R² (all components, computed on $set_name):")
    println("  Original (gyro·code): $(round(r2_orig, digits=4))")
    println("  Processor: $(round(r2_proc, digits=4))")
    println("\nR² (non-sparse components only, threshold=$epsilon, computed on $set_name):")
    println("  Original (gyro·code): $(round(r2_orig_nonsparse, digits=4))")
    println("  Processor: $(round(r2_proc_nonsparse, digits=4))")
    println("\nAvg nonzero gyro components per datapoint on $set_name (|gyro| >= $epsilon, shape: $gyro_shape, total: $components_per_datapt):")
    println("  Original: $(round(Int, gyro_nonzero_per_datapt)) ($(round(100*gyro_nonzero_per_datapt/components_per_datapt, digits=2))%)")
    println("  Processor: $(round(Int, proc_gyro_nonzero_per_datapt)) ($(round(100*proc_gyro_nonzero_per_datapt/components_per_datapt, digits=2))%)")
    println("\nAvg nonzero gyro·code product components per datapoint on $set_name (|gyro·code| >= $epsilon):")
    println("  Original: $(round(Int, gyroprod_nonzero_per_datapt)) ($(round(100*gyroprod_nonzero_per_datapt/components_per_datapt, digits=2))%)")
    println("  Processor: $(round(Int, procprod_nonzero_per_datapt)) ($(round(100*procprod_nonzero_per_datapt/components_per_datapt, digits=2))%)")
    
    return ProcessorEvalStats(
        r2_orig, r2_proc,
        r2_orig_nonsparse, r2_proc_nonsparse,
        gyro_nonzero_per_datapt, proc_gyro_nonzero_per_datapt,
        gyroprod_nonzero_per_datapt, procprod_nonzero_per_datapt,
        components_per_datapt, gyro_shape
    )
end
