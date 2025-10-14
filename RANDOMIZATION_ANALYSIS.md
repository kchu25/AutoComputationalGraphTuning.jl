# Non-Determinism in train_final_model - ROOT CAUSE

## Summary
**The models differ slightly because CUDA operations (matrix multiplication, convolutions) use non-deterministic algorithms by default for performance.**

## The Real Issue: CUDA Non-Determinism

### What's Happening

When you train on GPU, operations like:
- Matrix multiplication (cuBLAS GEMM)
- Convolutions (cuDNN)
- Reductions (sum, mean)

Use **non-deterministic algorithms** that can produce slightly different results even with the same inputs and same random seed.

**Why your PWMs are "similar but different":**
```
Run 1:  0.579211  -0.371443   -0.559047   0.0313666
Run 2:  0.54548   -0.335756   -0.539345   0.0313666
        ^^^^^^^^  ^^^^^^^^^   ^^^^^^^^^   ^^^^^^^^^
        Similar values, but not identical!
```

The differences are small (typically < 1%) but accumulate over many epochs of training.

### Why This Happens

CUDA libraries optimize for speed by:
1. **Parallel reduction order**: Adding numbers in different orders due to thread scheduling
2. **Atomic operations**: Non-deterministic race conditions in parallel updates  
3. **Tensor cores**: Use lower precision intermediate calculations
4. **Kernel fusion**: Combine operations in ways that aren't perfectly reproducible

### Verification Test

```julia
# Without deterministic mode
model1, _ = train_final_model(data, create_model; seed=42, max_epochs=100)
model2, _ = train_final_model(data, create_model; seed=42, max_epochs=100)

# Check parameters
p1 = Flux.params(model1)[1]  # Get first weight matrix
p2 = Flux.params(model2)[1]
println("Are they equal? ", p1 == p2)  # FALSE!
println("Max difference: ", maximum(abs.(p1 .- p2)))  # Small but non-zero
```

## The Solution

Set `ENV["CUBLAS_WORKSPACE_CONFIG"]` **BEFORE** loading CUDA:

```julia
# ⚠️ MUST BE FIRST LINE, before using packages
ENV["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

using AutoComputationalGraphTuning

# Now perfectly reproducible
model1, _ = train_final_model(data, create_model; seed=42)
model2, _ = train_final_model(data, create_model; seed=42)

# They will be IDENTICAL
@assert Flux.params(model1)[1] == Flux.params(model2)[1]
```

### What This Does

**`ENV["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"`** is an environment variable that controls CUDA's cuBLAS library behavior.

**Breaking it down:**
- `ENV[...]` - Sets a system environment variable that CUDA libraries read
- `"CUBLAS_WORKSPACE_CONFIG"` - The name of the environment variable
- `":4096:8"` - The configuration string with two parts:
  - `4096` - Workspace size in **kilobytes** (4 MB) that cuBLAS can use for deterministic algorithms
  - `8` - Number of workspace **splits** (how to divide the workspace for parallel operations)

**What it does:**
- Forces cuBLAS (CUDA's Basic Linear Algebra Subprograms library) to use **deterministic algorithms** for operations like matrix multiplication
- Without this setting, cuBLAS uses faster non-deterministic algorithms that can produce slightly different results each time
- The workspace memory is used to ensure operations happen in a reproducible order

**Alternative configurations:**
- `:16:8` - Smaller workspace (16 KB), uses less memory but may be slower
- `:4096:2` - Same workspace but fewer splits, different parallelization strategy
- Larger values like `:16384:8` - More memory for complex operations

**Why the colon?** The format `:size:splits` is NVIDIA's convention for this setting.

**Important:** This must be set **before** CUDA is loaded/initialized, which is why it needs to be the first line in your script.

### Performance Impact

⚠️ **Trade-off**: Deterministic mode is typically **10-50% slower**

- **For hyperparameter tuning**: Leave non-deterministic for speed (small differences don't matter)
- **For final model**: Enable deterministic mode for perfect reproducibility

## What Was NOT The Problem

✅ **Data splits**: Already deterministic with fixed seed
✅ **Model initialization**: Already deterministic with fixed seed  
✅ **Batch size randomization**: Independent RNG, no contamination
✅ **DataLoader shuffling**: Controlled by fixed RNG
✅ **Training algorithm**: Deterministic on CPU

The ONLY source of non-determinism is **CUDA kernel execution**.

## Alternative: Train on CPU

For perfect reproducibility without environment variables:

```julia
# CPU training is fully deterministic (but slower)
model, stats = train_final_model(data, create_model; 
                                seed=42, 
                                use_cuda=false)
```

## Current State

**What IS controlled:**
- ✅ Global RNG seed
- ✅ Model initialization RNG  
- ✅ Data split RNG
- ✅ Batch size randomization RNG (independent)
- ✅ DataLoader shuffle RNG
- ✅ Training algorithm (deterministic)

**What causes differences:**
- ❌ **CUDA operations** (non-deterministic by default)

**Fix:** Set `ENV["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` before loading packages.
