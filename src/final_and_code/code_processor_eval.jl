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
    epsilon_used::T
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
    gyro_nonsparse_mask = abs.(proc_gyro) .>= epsilon
    
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
Find optimal epsilon threshold based on proc_gyro sparsity percentage.

Uses binary search to find epsilon where a target percentage of proc_gyro components
are above the threshold (non-sparse).
"""
function _find_optimal_epsilon(all_proc_gyro::AbstractVector{T};
                              target_sparsity_pct=0.999,
                              epsilon_min=1f-6, epsilon_max=5f-3,
                              max_iterations=20) where T<:AbstractFloat
    
    # Binary search for epsilon that achieves target sparsity
    eps_low, eps_high = T(epsilon_min), T(epsilon_max)
    best_epsilon = T(1f-3)  # Default fallback
    
    for _ in 1:max_iterations
        eps_mid = (eps_low + eps_high) / 2
        
        # Count components above threshold
        num_above = sum(abs.(all_proc_gyro) .>= eps_mid)
        current_sparsity = 1.0 - (num_above / length(all_proc_gyro))
        
        if abs(current_sparsity - target_sparsity_pct) < 0.01  # Within 1%
            best_epsilon = eps_mid
            break
        elseif current_sparsity < target_sparsity_pct
            # Too many components above threshold, increase epsilon
            eps_low = eps_mid
        else
            # Too few components above threshold, decrease epsilon
            eps_high = eps_mid
        end
        
        best_epsilon = eps_mid
    end
    
    return best_epsilon
end

"""
Evaluate code processor performance on a dataset.

# Arguments
- `model`: Trained model
- `processor`: Trained code processor
- `dataloader`: DataLoader for evaluation
- `set_name`: Name of the dataset (e.g., "Train", "Test")
- `epsilon`: Threshold for sparsity. If nothing, uses default or auto-finds.
- `auto_epsilon`: If true and epsilon=nothing, finds optimal epsilon automatically
- `target_sparsity_pct`: Target sparsity percentage for auto epsilon (default: 0.95)
- `predict_position`: Position for prediction (default: 1)

# Returns
`ProcessorEvalStats` containing R² scores, sparsity metrics, and epsilon used

# Example
```julia
# First call finds optimal epsilon on train set
stats_train = evaluate_processor(m, processor, dl_train, "Train"; auto_epsilon=true)
# Use same epsilon for test set
stats_test = evaluate_processor(m, processor, dl_test, "Test"; epsilon=stats_train.epsilon_used)
```
"""
function evaluate_processor(model, processor, dataloader, set_name::String;
                           epsilon::Union{Nothing, AbstractFloat}=nothing,
                           auto_epsilon::Bool=false,
                           target_sparsity_pct::AbstractFloat=0.999f0,
                           predict_position::Int=1)
    
    T = DEFAULT_FLOAT_TYPE
    
    # First pass: collect all data
    preds_collection = T[]
    gyro_prods_collection = T[]
    proc_prods_collection = T[]
    all_gyro = T[]
    all_proc_gyro = T[]
    all_gyro_code_products = T[]
    all_proc_gyro_code_products = T[]
    
    gyro_shape = nothing
    
    for (seq, _) in dataloader
        code = model.code(seq |> gpu)
        preds, gyro = _compute_gyro_and_preds(model, code, predict_position)
        
        if isnothing(gyro_shape)
            gyro_shape = size(gyro)
        end

        processor.training[] = false
        proc_gyro = processor(code, gyro)
        
        # Compute products
        gyro_code_product = gyro .* code
        proc_gyro_code_product = proc_gyro .* code
        
        gyro_prod = vec(sum(gyro_code_product, dims=(1,2)))
        proc_prod = vec(sum(proc_gyro_code_product, dims=(1,2)))
        
        # Collect data
        append!(preds_collection, vec(cpu(preds)))
        append!(gyro_prods_collection, cpu(gyro_prod))
        append!(proc_prods_collection, cpu(proc_prod))
        append!(all_gyro, vec(cpu(gyro)))
        append!(all_proc_gyro, vec(cpu(proc_gyro)))
        append!(all_gyro_code_products, vec(cpu(gyro_code_product)))
        append!(all_proc_gyro_code_products, vec(cpu(proc_gyro_code_product)))
    end
    
    # Determine epsilon
    if isnothing(epsilon) && auto_epsilon
        println("Finding optimal epsilon on $set_name set (target sparsity: $(round(100*target_sparsity_pct, digits=1))%)...")
        epsilon = _find_optimal_epsilon(all_proc_gyro; target_sparsity_pct=target_sparsity_pct)
        println("Optimal epsilon found: $epsilon")
    elseif isnothing(epsilon)
        epsilon = T(5f-3)  # Default
    else
        epsilon = T(epsilon)  # Use provided
    end
    
    # Second pass: compute statistics with determined epsilon
    preds_all = preds_collection
    gyro_prods = gyro_prods_collection
    proc_prods = proc_prods_collection
    
    # Count sparsity
    gyro_sparse_count = sum(abs.(all_gyro) .< epsilon)
    proc_sparse_count = sum(abs.(all_proc_gyro) .< epsilon)
    gyroprod_sparse_count = sum(abs.(all_gyro_code_products) .< epsilon)
    procprod_sparse_count = sum(abs.(all_proc_gyro_code_products) .< epsilon)
    
    # Compute non-sparse predictions (masked by proc_gyro)
    gyro_prods_nonsparse = T[]
    proc_prods_nonsparse = T[]
    
    components_per_sample = prod(gyro_shape[1:3])
    num_samples = length(preds_all)
    
    for i in 1:num_samples
        gyro_sum_nonsparse = zero(T)
        proc_sum_nonsparse = zero(T)
        start_idx = (i-1) * components_per_sample + 1
        end_idx = i * components_per_sample
        
        for j in start_idx:end_idx
            if abs(all_proc_gyro[j]) >= epsilon  # Mask based on proc_gyro
                gyro_sum_nonsparse += all_gyro_code_products[j]
                proc_sum_nonsparse += all_proc_gyro_code_products[j]
            end
        end
        push!(gyro_prods_nonsparse, gyro_sum_nonsparse)
        push!(proc_prods_nonsparse, proc_sum_nonsparse)
    end
    
    # Compute R² scores (all components)
    r2_orig = _compute_r2(preds_all, gyro_prods)
    r2_proc = _compute_r2(preds_all, proc_prods)
    
    # Compute R² scores (non-sparse components only)
    r2_orig_nonsparse = _compute_r2(preds_all, gyro_prods_nonsparse)
    r2_proc_nonsparse = _compute_r2(preds_all, proc_prods_nonsparse)
    
    # Compute per-datapoint sparsity metrics
    components_per_datapt = components_per_sample
    gyro_total_count = length(all_gyro)
    num_datapoints = num_samples
    
    gyro_nonzero_per_datapt = T((gyro_total_count - gyro_sparse_count) / num_datapoints)
    proc_gyro_nonzero_per_datapt = T((gyro_total_count - proc_sparse_count) / num_datapoints)
    
    gyroprod_nonzero_per_datapt = T((gyro_total_count - gyroprod_sparse_count) / num_datapoints)
    procprod_nonzero_per_datapt = T((gyro_total_count - procprod_sparse_count) / num_datapoints)
    
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
        components_per_datapt, gyro_shape,
        epsilon
    )
end
