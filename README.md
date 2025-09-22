# AutoComputationalGraphTuning

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kchu25.github.io/AutoComputationalGraphTuning.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kchu25.github.io/AutoComputationalGraphTuning.jl/dev/)
[![Build Status](https://github.com/kchu25/AutoComputationalGraphTuning.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kchu25/AutoComputationalGraphTuning.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/kchu25/AutoComputationalGraphTuning.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/kchu25/AutoComputationalGraphTuning.jl)


This package just requires two things, one is the struct `data` and another is a function to create a Flux model `model` (using Flux.jl). 
Following the duck typing practices, these must satisfy the following:

- `data`
    - four fields are required"
    - `data.X` returns the features
    - `data.Y` returns the labels
    - `data.X_dim` returns the dimension of each feature
    - `data.Y_dim` returns the dimension of each label


- `model`
    - must define a function called `create_model`

