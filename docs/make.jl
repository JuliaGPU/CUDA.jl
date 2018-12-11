using Documenter, CuArrays

makedocs(
    modules = [CuArrays],
    format = Documenter.HTML(),
    sitename = "CuArrays.jl",
    pages = [
        "Home" => "index.md",
    ],
    doctest = true
)
