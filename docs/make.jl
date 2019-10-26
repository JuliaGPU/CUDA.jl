using Documenter, Literate
using CUDAapi, CUDAdrv, CUDAnative, CuArrays

@info "Building Literate.jl documentation"
Literate.markdown("src/tutorials/introduction.jl", "src/tutorials")

@info "Generating Documenter.jl site"
makedocs(
    modules = [CUDAapi, CUDAdrv, CUDAnative, CuArrays],
    doctest = false,
    clean = true,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    sitename = "Julia/CUDA",
    authors = "Tim Besard",
    pages = Any[
        "Home" => "index.md",
        "Tutorials" => Any[
            "tutorials/introduction.md",
        ],
        "Usage" => Any[
            "usage/overview.md",
            "usage/faq.md",
        ],
        "Development" => Any[
            "development/workflow.md",
        ],
      ]
)
