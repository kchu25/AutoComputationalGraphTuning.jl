# Demonstration of RNG Contamination Bug

## The Bug

When `randomize_batchsize=true`, the batch size selection was contaminating the global RNG:

```julia
function _setup_final_trial(raw_data, create_model, seed; randomize_batchsize, ...)
    rng_global = set_reproducible_seeds!(seed)  # RNG at state S0
    
    # BUG: This advances rng_global!
    batch_size = randomize_batchsize ? rand(rng_global, BATCH_SIZE_RANGE) : DEFAULT_BATCH_SIZE
    
    # Now rng_global is at state S1, not S0!
    setup = setup_model_and_training_final(..., rng=rng_global, ...)
end
```

Inside `setup_model_and_training_final`:
```julia
model_rng = MersenneTwister(rand(rng, UInt))  # Uses contaminated rng!
split_rng = MersenneTwister(rand(rng, UInt))  # Also contaminated!
```

## Why This Causes Non-Determinism

**First run with seed=42:**
1. `rng_global` set to seed 42 (state S0)
2. `rand(rng_global, BATCH_SIZE_RANGE)` → returns batch_size=128, RNG advances to S1
3. `model_rng` created with `rand(S1)` → gets seed X1
4. Model initialized with seed X1 → weights W1

**Second run with seed=42:**
1. `rng_global` set to seed 42 (state S0) — SAME!
2. `rand(rng_global, BATCH_SIZE_RANGE)` → returns batch_size=128, RNG advances to S1 — SAME!
3. `model_rng` created with `rand(S1)` → gets seed X1 — SAME!
4. Model initialized with seed X1 → weights W1 — **SHOULD BE SAME!**

Wait... this should actually be deterministic! Let me think harder...

## The ACTUAL Bug

The issue is that `randomize_batchsize=true` means different batch sizes might be chosen, and:

1. **Different batch sizes lead to different training dynamics**
2. Even if the model initialization is the same, **different batch sizes → different gradients**

BUT if you're using the same seed and getting different results, there must be something else...

## Alternative Theory: DataLoader Shuffle

Look at this code:
```julia
dl_combined = Flux.DataLoader(
    (setup.processed_data.train.tensor, setup.processed_data.train.labels),
    batchsize = setup.batch_size,
    shuffle = true,  # ← SHUFFLES EVERY EPOCH
    partial = false,
    rng = MersenneTwister(seed)
)
```

The `MersenneTwister(seed)` RNG is created fresh, but it's **reused across epochs**:
- Epoch 1: Shuffle with RNG at state S0
- Epoch 2: Shuffle with RNG at state S1 (after epoch 1 shuffle)
- Epoch 3: Shuffle with RNG at state S2
- ...

This means if you train for different numbers of epochs or if early stopping triggers at different points, you get different shuffle patterns!

## The Fix (Already Applied)

Use a separate RNG for batch size selection that doesn't contaminate the main RNG:

```julia
function _setup_final_trial(raw_data, create_model, seed; randomize_batchsize, ...)
    rng_global = set_reproducible_seeds!(seed)
    
    # Use SEPARATE RNG for batch size
    batch_size_rng = MersenneTwister(seed + 999999)
    batch_size = randomize_batchsize ? rand(batch_size_rng, BATCH_SIZE_RANGE) : DEFAULT_BATCH_SIZE
    
    # Now rng_global is CLEAN and deterministic
    setup = setup_model_and_training_final(..., rng=rng_global, ...)
end
```

## Verification

Test if the fix works:

```julia
# Run 1
model1, stats1 = train_final_model(data, create_model; 
                                   seed=42, 
                                   randomize_batchsize=true,
                                   use_cuda=false,
                                   max_epochs=50)

# Run 2 (same seed, same params)
model2, stats2 = train_final_model(data, create_model; 
                                   seed=42, 
                                   randomize_batchsize=true,
                                   use_cuda=false,
                                   max_epochs=50)

# Check if identical
using Flux
p1 = Flux.params(model1)
p2 = Flux.params(model2)

for (param1, param2) in zip(p1, p2)
    if param1 != param2
        println("❌ DIFFERENT!")
        println("Max diff: ", maximum(abs.(param1 .- param2)))
    else
        println("✅ IDENTICAL!")
    end
end
```

If you still see differences after this fix, the issue is likely:
1. DataLoader shuffle across epochs (stateful RNG)
2. Some other code using Random.GLOBAL_RNG between calls
3. Flux/Julia version differences in how operations are computed
