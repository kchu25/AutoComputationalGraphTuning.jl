# CUDA Determinism Setup for Reproducible Training

## The Problem
CUDA operations (matrix multiplication, convolutions, etc.) use non-deterministic algorithms by default for better performance. This causes models to differ slightly even with the same random seed.

## The Complete Solution

You need **MULTIPLE** environment variables set **BEFORE** starting Julia:

### Option 1: Set in shell before running Julia (RECOMMENDED)

```bash
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_LAUNCH_BLOCKING="1"
julia
```

### Option 2: Set at the VERY START of your script (before any imports)

```julia
# ⚠️ MUST BE FIRST LINES, before using ANY packages
ENV["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
ENV["CUDA_LAUNCH_BLOCKING"] = "1"

# Now load packages
using Pkg
Pkg.activate(".")
using AutoComputationalGraphTuning
# ... rest of your code
```

### Option 3: Set in Julia startup file

Add to `~/.julia/config/startup.jl`:

```julia
ENV["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
ENV["CUDA_LAUNCH_BLOCKING"] = "1"
```

## What Each Setting Does

### `CUBLAS_WORKSPACE_CONFIG = ":4096:8"`
Forces cuBLAS (CUDA's matrix multiplication library) to use deterministic algorithms.
- **Without it**: Matrix operations use fast but non-deterministic parallel reductions
- **With it**: Uses workspace memory to ensure consistent computation order
- See detailed explanation below

### `CUDA_LAUNCH_BLOCKING = "1"`  
Forces CUDA kernels to run synchronously (one at a time).
- **Without it**: Multiple CUDA kernels can run concurrently in unpredictable order
- **With it**: Kernels execute in strict sequential order
- **Critical for reproducibility** across runs

## What this does

`CUBLAS_WORKSPACE_CONFIG` forces cuBLAS (CUDA's BLAS library) to use deterministic algorithms.

### Understanding the configuration string

**Format:** `:workspace_size_kb:num_splits`

**`:4096:8`** means:
- **4096** = Workspace size in kilobytes (4 MB of GPU memory)
  - This memory is used as scratch space for deterministic operations
  - Larger workspace → can handle bigger matrices deterministically
  - But uses more GPU memory
  
- **8** = Number of workspace splits
  - How the workspace is divided for parallel operations
  - More splits → better parallelization but more coordination overhead
  - Fewer splits → simpler but may be slower

**Common configurations:**
- `:4096:8` - **Recommended** - Good balance of memory and performance
- `:16:8` - Minimal memory (16 KB), slower but uses almost no extra GPU RAM
- `:16384:8` - Large workspace (16 MB), for very large matrices
- `:4096:2` - Same memory but fewer splits, different parallelization

### How it works

**Without deterministic mode (default):**
```
Matrix multiplication: A × B = C
- Thread 1: Computes C[0,0] = A[0,:] · B[:,0]
- Thread 2: Computes C[0,1] = A[0,:] · B[:,1]
- Threads run in parallel, order varies → non-deterministic!
```

**With deterministic mode:**
```
Matrix multiplication: A × B = C
- Uses workspace to ensure threads execute in fixed order
- Same computation order every time → deterministic!
- But requires synchronization → slower
```

### Why this matters

CUDA parallelizes operations across thousands of threads. The order these threads complete isn't guaranteed, leading to:
- Floating-point operations in different orders
- Slightly different rounding errors
- Results that differ in the last few bits

The workspace ensures a consistent execution order.

## Performance Impact

⚠️ **Warning**: Deterministic mode is typically **10-50% slower** than non-deterministic mode.
- For tuning (many trials): Consider leaving non-deterministic for speed
- For final model: Enable deterministic mode for perfect reproducibility

## Verification

After setting the environment variable, test reproducibility:

```julia
ENV["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Must be FIRST

using AutoComputationalGraphTuning

# Train twice with same seed
model1, stats1 = train_final_model(data, create_model; seed=42)
model2, stats2 = train_final_model(data, create_model; seed=42)

# Check if identical
using Flux
params1 = Flux.params(model1)
params2 = Flux.params(model2)

all_equal = true
for (p1, p2) in zip(params1, params2)
    if p1 != p2
        println("Difference found!")
        println("Max diff: ", maximum(abs.(p1 .- p2)))
        all_equal = false
    end
end

if all_equal
    println("✅ Perfect reproducibility!")
else
    println("❌ Still some differences - may need additional CUDA settings")
end
```

## Additional Settings (If Still Non-Deterministic)

If you still see differences, try these additional flags:

```bash
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_LAUNCH_BLOCKING="1"
export CUDNN_DETERMINISTIC="1"  # Force deterministic cuDNN operations
julia
```

Or in Julia:
```julia
ENV["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
ENV["CUDA_LAUNCH_BLOCKING"] = "1"  
ENV["CUDNN_DETERMINISTIC"] = "1"

using AutoComputationalGraphTuning
```

## CPU-only Alternative

If perfect reproducibility is critical and you don't need GPU speed:

```julia
# Train on CPU instead
model, stats = train_final_model(data, create_model; seed=42, use_cuda=false)
```

CPU operations are fully deterministic (but slower).
