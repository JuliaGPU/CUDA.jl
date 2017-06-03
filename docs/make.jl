using Documenter, CUDAdrv

makedocs(
    modules = [CUDAdrv],
    format = :html,
    sitename = "CUDAdrv.jl",
    pages = [
        "Home" => "index.md",
        "Manual" => [],
        "Library" => ["api.md", "array.md"]
    ]
)

deploydocs(
    repo = "github.com/JuliaGPU/CUDAdrv.jl.git",
    target = "build",
    deps = nothing,
    make = nothing
)
