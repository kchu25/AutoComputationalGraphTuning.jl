#!/usr/bin/env julia

# Test script for CUDA determinism
# This script trains two models with the same seed and checks if they're identical

println("="^70)
println("CUDA Determinism Test")
println("="^70)

# CRITICAL: Set these BEFORE loading any packages
ENV["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
ENV["CUDA_LAUNCH_BLOCKING"] = "1"

println("\n✓ Environment variables set:")
println("  CUBLAS_WORKSPACE_CONFIG = ", get(ENV, "CUBLAS_WORKSPACE_CONFIG", "NOT SET"))
println("  CUDA_LAUNCH_BLOCKING = ", get(ENV, "CUDA_LAUNCH_BLOCKING", "NOT SET"))

# Now load packages
using Pkg
Pkg.activate(".")
using AutoComputationalGraphTuning
using Flux

println("\n" * "="^70)
println("Training Model 1...")
println("="^70)

# Train first model
model1, stats1, setup1 = train_final_model(
    your_data,  # Replace with your actual data
    your_create_model_function,  # Replace with your model creation function
    seed=42,
    max_epochs=50,
    use_cuda=true,
    randomize_batchsize=false  # Use fixed batch size for testing
)

println("\n" * "="^70)
println("Training Model 2...")
println("="^70)

# Train second model with SAME seed
model2, stats2, setup2 = train_final_model(
    your_data,  # Same data
    your_create_model_function,  # Same model function
    seed=42,  # SAME SEED
    max_epochs=50,
    use_cuda=true,
    randomize_batchsize=false
)

println("\n" * "="^70)
println("Checking Determinism...")
println("="^70)

# Compare parameters
params1 = Flux.params(model1)
params2 = Flux.params(model2)

all_identical = true
total_params = 0
max_diff = 0.0

for (i, (p1, p2)) in enumerate(zip(params1, params2))
    total_params += length(p1)
    
    if p1 == p2
        println("✅ Parameter set $i: IDENTICAL")
    else
        all_identical = false
        diff = maximum(abs.(p1 .- p2))
        max_diff = max(max_diff, diff)
        println("❌ Parameter set $i: DIFFERENT (max diff = $diff)")
        
        # Show a sample of differences
        if length(p1) >= 16
            println("   Sample from parameter set $i:")
            println("   Model 1: ", p1[1:min(4,end)])
            println("   Model 2: ", p2[1:min(4,end)])
        end
    end
end

println("\n" * "="^70)
println("SUMMARY")
println("="^70)
println("Total parameters: $total_params")

if all_identical
    println("\n🎉 SUCCESS: Models are PERFECTLY IDENTICAL!")
    println("✓ Deterministic mode is working correctly")
else
    println("\n❌ FAILURE: Models are DIFFERENT")
    println("✗ Maximum difference: $max_diff")
    println("\nPossible causes:")
    println("  1. Environment variables not set before loading CUDA")
    println("  2. Additional CUDA settings needed (try CUDNN_DETERMINISTIC=1)")
    println("  3. Some operations in your model may not be deterministic")
    println("\nTo fix:")
    println("  - Restart Julia completely")
    println("  - Set ENV variables FIRST, before any 'using' statements")
    println("  - See CUDA_DETERMINISM.md for more details")
end

println("="^70)
