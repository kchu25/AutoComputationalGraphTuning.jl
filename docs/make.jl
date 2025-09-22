using AutoComputationalGraphTuning
using Documenter

DocMeta.setdocmeta!(AutoComputationalGraphTuning, :DocTestSetup, :(using AutoComputationalGraphTuning); recursive=true)

makedocs(;
    modules=[AutoComputationalGraphTuning],
    authors="Shane Kuei-Hsien Chu (skchu@wustl.edu)",
    sitename="AutoComputationalGraphTuning.jl",
    format=Documenter.HTML(;
        canonical="https://kchu25.github.io/AutoComputationalGraphTuning.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/kchu25/AutoComputationalGraphTuning.jl",
    devbranch="main",
)
