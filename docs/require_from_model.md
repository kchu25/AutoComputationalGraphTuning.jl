# Required Interface from `model`

This document lists all fields and methods that `AutoComputationalGraphTuning` expects
from the user-supplied model. Any model passed to this package **must** implement all of
these to avoid runtime errors.

---

## 1. Callable — `model(x)`

The model itself must be callable (i.e., a Flux `Chain` or custom functor).

| Usage | File(s) |
|-------|---------|
| `model(seq)` | `_training/train.jl`, `_training/eval.jl` |

Used during the main training loop and validation evaluation to produce predictions.

---

## 2. `model.code(x)` — Code Layer

A sub-layer that maps raw input sequences to a learned "code" representation.

| Usage | File(s) |
|-------|---------|
| `model.code(seq)` | `_training/customized_losses.jl`, `final_and_code/_helpers.jl`, `final_and_code/code_processor_eval.jl`, `final_and_code/gyro_thresh.jl` |

---

## 3. `model.predict_up_to_final_nonlinearity` — Linear Prediction Layer

A callable layer that takes code and returns predictions **before** the final
nonlinearity (e.g., sigmoid). Called with keyword `predict_position`.

| Usage | File(s) |
|-------|---------|
| `linear_sum_fcn = @ignore model.predict_up_to_final_nonlinearity` | `final_and_code/_helpers.jl`, `final_and_code/code_processor_eval.jl`, `final_and_code/gyro_thresh.jl` |
| `predict_up_to_final_nonlinearity(model, code; predict_position=...)` | `_training/customized_losses.jl` (passed as a function argument) |

**Signature:** `model.predict_up_to_final_nonlinearity(code; predict_position=Int)`

---

## 4. `model.final_nonlinearity` — Output Activation

The final element-wise activation function (e.g., `sigmoid`, `identity`).

| Usage | File(s) |
|-------|---------|
| `model.final_nonlinearity.(predict_upto_fn)` | `_training/customized_losses.jl` |

Applied element-wise (broadcasted `.()`) to the output of
`predict_up_to_final_nonlinearity`.

---

## 5. `model.training` — Training/Eval Mode Flag (`Ref{Bool}`)

A `Ref{Bool}` toggled to switch the model between training and evaluation mode.

| Usage | File(s) |
|-------|---------|
| `model.training[] = false` | `_training/train.jl` (eval mode for validation) |
| `model.training[] = true` | `_training/train.jl` (back to training mode) |

---

## 6. `model.hp` — Hyperparameters Struct

A struct (or named tuple) containing model hyperparameters. Used by the code
processor training pipeline.

| Field accessed | Usage | File(s) |
|----------------|-------|---------|
| `model.hp.inference_code_layer` | Default layer index for code inference | `final_and_code/train_code_processor.jl` |
| `model.hp` (whole struct) | Passed to `_init_processor(proc_wrap, model.hp, ...)` | `final_and_code/train_code_processor.jl` |

---

## Summary Checklist

```
model              # callable: model(x) → predictions
├── .code          # callable: model.code(x) → code representation
├── .predict_up_to_final_nonlinearity
│                  # callable: (code; predict_position) → pre-activation predictions
├── .final_nonlinearity
│                  # element-wise function: e.g. sigmoid, identity
├── .training      # Ref{Bool}: training/eval mode flag
└── .hp            # struct with at least:
    └── .inference_code_layer  # Int: layer index for code inference
```

> **Tip:** If adding a new loss or evaluation function, make sure it only accesses
> fields listed above. Any new model field should be documented here.
