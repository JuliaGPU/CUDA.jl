using Documenter, CUDAnative

makedocs(
    modules = [CUDAnative],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    sitename = "CUDAnative.jl",
    pages = [
        "Home"    => "index.md",
        "Manual"  => [
            "man/usage.md",
            "man/troubleshooting.md",
            "man/performance.md",
            "man/hacking.md"
        ],
        "Library" => [
            "lib/compilation.md",
            "lib/reflection.md",
            "lib/profiling.md",
            "Device Code" => [
                "lib/device/intrinsics.md",
                "lib/device/array.md",
                "lib/device/libdevice.md"
            ]
        ]
    ],
    doctest = true
)
