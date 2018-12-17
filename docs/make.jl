using Documenter, CUDAdrv

makedocs(
    modules = [CUDAdrv],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    sitename = "CUDAdrv.jl",
    pages = [
        "Home"    => "index.md",
        "Manual"  => [
            "man/usage.md"
        ],
        "Library" => [
            "lib/api.md",
        ]
    ],
    doctest = true
)
