using Documenter, CuArrays

makedocs(
    modules = [CuArrays],
    format = :html,
    sitename = "CuArrays.jl",
    pages = [
        "Home"    => "index.md",
    ],
    doctest = true
)

deploydocs(
    repo = "github.com/JuliaGPU/CuArrays.jl.git",
    julia = "",
    osname = "",
    # no need to build anything here, re-use output of `makedocs`
    target = "build",
    deps = nothing,
    make = nothing
)
