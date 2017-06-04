using Documenter, CUDAnative

const test = haskey(ENV, "TEST")    # are we running as part of the test suite?

makedocs(
    modules = [CUDAnative],
    format = :html,
    sitename = "CUDAnative.jl",
    pages = [
        "Home"    => "index.md",
        "Manual"  => [
            "man/usage.md",
            "man/troubleshooting.md",
            "man/performance.md"
        ],
        "Library" => [
            "lib/intrinsics.md"
        ]
    ],
    doctest = test
)

test || deploydocs(
    repo = "github.com/JuliaGPU/CUDAnative.jl.git",
    julia = "0.6",
    # no need to build anything here, re-use output of `makedocs`
    target = "build",
    deps = nothing,
    make = nothing
)
