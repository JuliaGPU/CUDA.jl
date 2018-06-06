using Documenter, CUDAdrv

makedocs(
    modules = [CUDAdrv],
    format = :html,
    sitename = "CUDAdrv.jl",
    pages = [
        "Home"    => "index.md",
        "Manual"  => [
            "man/usage.md"
        ],
        "Library" => [
            "lib/api.md",
            "lib/array.md"
        ]
    ],
    doctest = true
)

deploydocs(
    repo = "github.com/JuliaGPU/CUDAdrv.jl.git",
    julia = "0.6",
    # no need to build anything here, re-use output of `makedocs`
    target = "build",
    deps = nothing,
    make = nothing
)
