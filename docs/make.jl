using Documenter, CuArrays

makedocs(
    modules = [CuArrays],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    sitename = "CuArrays.jl",
    pages = [
        "Home" => "index.md",
    ],
    doctest = true
)
