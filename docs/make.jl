using Documenter
using Literate

using CuArrays

# generate tutorials
OUTPUT = joinpath(@__DIR__, "src/tutorials/generated")
Literate.markdown(joinpath(@__DIR__, "src/tutorials/intro.jl"), OUTPUT)

makedocs(
    modules = [CuArrays],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    sitename = "CuArrays.jl",
    pages = [
        "Home" => "index.md",
        "Tutorials"  => [
            "tutorials/generated/intro.md"
        ],
    ],
    doctest = true
)
