using Documenter, CuArrays

using Pkg
if haskey(ENV, "GITLAB_CI")
    Pkg.add([PackageSpec(name = x; rev = "master")
             for x in ["CUDAapi", "GPUArrays", "CUDAnative", "NNlib", "CUDAdrv"]])
end

makedocs(
    modules = [CuArrays],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    sitename = "CuArrays.jl",
    pages = [
        "Home" => "index.md",
    ],
    doctest = true
)
