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
