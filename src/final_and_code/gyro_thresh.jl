"""Statistics for threshold evaluation"""
struct ThresholdEvalStats{T<:AbstractFloat}
    threshold::T
    r2_original::T
    r2_processor::T
    sparsity_pct::T  # Percentage of components zeroed out
    avg_nonzero_per_sample::T  # Average number of non-zero components per sample
    std_nonzero_per_sample::T  # Standard deviation of non-zero components per sample
    min_nonzero_per_sample::Int  # Minimum non-zero components in any sample
    max_nonzero_per_sample::Int  # Maximum non-zero components in any sample
end

function Base.show(io::IO, stats::ThresholdEvalStats)
    println(io, "ThresholdEvalStats:")
    println(io, "  Threshold:            ", round(stats.threshold, sigdigits=4))
    println(io, "  R² Original:          ", round(stats.r2_original, digits=4))
    println(io, "  R² Processor:         ", round(stats.r2_processor, digits=4))
    println(io, "  Sparsity:             ", round(stats.sparsity_pct, digits=1), "%")
    println(io, "  Non-zero per sample:")
    println(io, "    Mean:               ", round(stats.avg_nonzero_per_sample, digits=1))
    println(io, "    Std:                ", round(stats.std_nonzero_per_sample, digits=1))
    println(io, "    Min:                ", stats.min_nonzero_per_sample)
    print(io, "    Max:                ", stats.max_nonzero_per_sample)
end

"""
Find optimal threshold for proc_gyro that maximizes sparsity while maintaining R² performance.

Searches for the highest threshold where masking code components below threshold
still maintains acceptable R² on the test set compared to the baseline processor R².

# Arguments
- `model`: Trained model
- `processor`: Trained code processor
- `dataloader_train`: Training data loader (for finding threshold)
- `dataloader_test`: Test data loader (for evaluating R²)
- `baseline_stats`: ProcessorEvalStats from evaluate_processor (contains baseline r2_processor)
- `r2_tolerance`: Maximum acceptable R² drop (default: 0.05, meaning max 5% relative drop)
- `num_candidates`: Number of threshold values to test (default: 20)
- `predict_position`: Position for prediction (default: 1)

# Returns
`ThresholdEvalStats` containing optimal threshold and performance metrics

# Example
```julia
# First get baseline performance
baseline = evaluate_processor(m, processor, dl_test, "Test")
# Then find optimal threshold
thresh_stats = find_optimal_threshold(m, processor, dl_train, dl_test, baseline)
```
"""
function find_optimal_threshold(model, processor, dataloader_train, dataloader_test,
                               baseline_stats::ProcessorEvalStats{T};
                               r2_tolerance::AbstractFloat=0.05f0,
                               num_candidates::Int=20,
                               predict_position::Int=1) where T<:AbstractFloat
    
    # First, collect code magnitudes from training set to determine threshold range
    println("\n=== Collecting code statistics from training set ===")
    all_code = T[]
    
    for (seq, _) in dataloader_train
        code = model.code(seq |> gpu)
        
        append!(all_code, abs.(vec(cpu(code))))
    end
    
    # Determine threshold candidates based on code distribution
    code_sorted = sort(all_code)
    min_thresh = quantile(code_sorted, 0.5)   # Start from median
    max_thresh = quantile(code_sorted, 0.999) # Up to 99.9th percentile
    
    threshold_candidates = T.(exp10.(range(log10(max(min_thresh, 1f-6)), 
                                           log10(max_thresh), 
                                           length=num_candidates)))
    
    println("Threshold range: $(minimum(threshold_candidates)) to $(maximum(threshold_candidates))")
    
    # Minimum acceptable R² on test set
    min_acceptable_r2 = baseline_stats.r2_processor * (1 - r2_tolerance)
    println("Baseline R² (processor): $(round(baseline_stats.r2_processor, digits=4))")
    println("Minimum acceptable R² (test): $(round(min_acceptable_r2, digits=4))")
    
    # Test each threshold on test set
    println("\n=== Testing thresholds on test set ===")
    best_threshold = T(0)
    best_stats = nothing
    
    for thresh in threshold_candidates
        # Evaluate on test set with this threshold
        stats = _evaluate_with_threshold(model, processor, dataloader_test, thresh, predict_position)
        
        println("Threshold: $(round(thresh, sigdigits=3)) -> " *
                "R² orig: $(round(stats.r2_original, digits=4)), " *
                "R² proc: $(round(stats.r2_processor, digits=4)), " *
                "Sparsity: $(round(stats.sparsity_pct, digits=1))%")
        
        # Keep the highest threshold that maintains acceptable R²
        if stats.r2_processor >= min_acceptable_r2 && thresh > best_threshold
            best_threshold = thresh
            best_stats = stats
        end
    end
    
    if isnothing(best_stats)
        @warn "No threshold found that maintains R² within tolerance. Using minimal threshold."
        best_stats = _evaluate_with_threshold(model, processor, dataloader_test, 
                                              minimum(threshold_candidates), predict_position)
    end
    
    println("\n=== Optimal Threshold Found ===")
    println("Threshold: $(round(best_stats.threshold, sigdigits=4))")
    println("R² (original): $(round(best_stats.r2_original, digits=4))")
    println("R² (processor with threshold): $(round(best_stats.r2_processor, digits=4))")
    println("Sparsity: $(round(best_stats.sparsity_pct, digits=1))% of components zeroed out")
    println("Non-zero components per sample:")
    println("  Mean: $(round(best_stats.avg_nonzero_per_sample, digits=1))")
    println("  Std:  $(round(best_stats.std_nonzero_per_sample, digits=1))")
    println("  Min:  $(best_stats.min_nonzero_per_sample)")
    println("  Max:  $(best_stats.max_nonzero_per_sample)")
    println("R² drop from baseline: $(round(100*(baseline_stats.r2_processor - best_stats.r2_processor)/baseline_stats.r2_processor, digits=2))%")
    
    return best_stats
end

"""
Evaluate processor performance with a specific threshold applied to code.

# Arguments
- `model`: Trained model
- `processor`: Trained code processor
- `dataloader`: DataLoader for evaluation
- `threshold`: Magnitude threshold for masking code components
- `predict_position`: Position for prediction

# Returns
`ThresholdEvalStats` containing R² scores and sparsity metrics
"""
function _evaluate_with_threshold(model, processor, dataloader, 
                                 threshold::T, predict_position::Int) where T<:AbstractFloat
    
    preds_collection = T[]
    gyro_prods_collection = T[]
    proc_prods_collection = T[]
    total_components = 0
    zeroed_components = 0
    num_samples = 0
    nonzero_counts_per_sample = Int[]
    
    for (seq, _) in dataloader
        code = model.code(seq |> gpu)
        
        # Apply threshold to code: zero out components below threshold
        mask = abs.(code) .>= threshold
        code_thresholded = code .* mask
        
        # Compute predictions and gradients using thresholded code
        (_, preds), gyro = Flux.withgradient(code_thresholded) do x
            linear_sum_fcn = @ignore model.predict_up_to_final_nonlinearity;
            preds = linear_sum_fcn(x; predict_position=predict_position)
            preds |> sum, preds
        end
        gyro = gyro[1]
        
        processor.training[] = false
        proc_gyro = processor(code_thresholded, gyro)
        
        # Track sparsity
        total_components += length(code)
        zeroed_components += sum(.!mask)
        
        # Track per-sample non-zero counts
        batch_size = size(code, 4)
        num_samples += batch_size
        components_per_sample = div(length(code), batch_size)
        
        # Reshape mask to count per sample
        mask_reshaped = reshape(mask, components_per_sample, batch_size)
        for i in 1:batch_size
            push!(nonzero_counts_per_sample, sum(mask_reshaped[:, i]))
        end
        
        # Compute products
        gyro_code_product = gyro .* code_thresholded
        proc_gyro_code_product = proc_gyro .* code_thresholded
        
        gyro_prod = vec(sum(gyro_code_product, dims=(1,2)))
        proc_prod = vec(sum(proc_gyro_code_product, dims=(1,2)))
        
        # Collect
        append!(preds_collection, vec(cpu(preds)))
        append!(gyro_prods_collection, cpu(gyro_prod))
        append!(proc_prods_collection, cpu(proc_prod))
    end
    
    # Compute R² scores
    r2_orig = _compute_r2(preds_collection, gyro_prods_collection)
    r2_proc = _compute_r2(preds_collection, proc_prods_collection)
    
    # Compute sparsity percentage and statistics
    sparsity_pct = T(100.0 * zeroed_components / total_components)
    avg_nonzero_per_sample = T(mean(nonzero_counts_per_sample))
    std_nonzero_per_sample = T(std(nonzero_counts_per_sample))
    min_nonzero = minimum(nonzero_counts_per_sample)
    max_nonzero = maximum(nonzero_counts_per_sample)
    
    return ThresholdEvalStats(threshold, r2_orig, r2_proc, sparsity_pct, 
                             avg_nonzero_per_sample, std_nonzero_per_sample,
                             min_nonzero, max_nonzero)
end
