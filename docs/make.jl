using Documenter, CUDAnative

makedocs(
    modules = [CUDAnative],
    format = :html,
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

deploydocs(
    repo = "github.com/JuliaGPU/CUDAnative.jl.git",
    julia = "0.6",
    # no need to build anything here, re-use output of `makedocs`
    target = "build",
    deps = nothing,
    make = nothing
)
