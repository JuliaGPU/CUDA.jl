using Documenter, Literate
using CUDAapi, CUDAdrv, CUDAnative, CuArrays

const src = "https://github.com/JuliaGPU/CUDA.jl"
const dst = "https://juliagpu.gitlab.io/CUDA.jl/"

function main()
    @info "Building Literate.jl documentation"
    cd(@__DIR__) do
        Literate.markdown("src/tutorials/introduction.jl", "src/tutorials";
                          repo_root_url="$src/blob/master/docs")
    end

    @info "Generating Documenter.jl site"
    makedocs(
        sitename = "CUDA.jl",
        authors = "Tim Besard",
        repo = "$src/blob/{commit}{path}#{line}",
        format = Documenter.HTML(
            # Use clean URLs on CI
            prettyurls = get(ENV, "CI", nothing) == "true",
            canonical = dst,
            assets = ["assets/favicon.ico"],
            analytics = "UA-154489943-2",
        ),
        doctest = false,
        pages = Any[
            "Home" => "index.md",
            "Tutorials" => Any[
                "tutorials/introduction.md",
            ],
            "Usage" => Any[
                "usage/overview.md",
                "usage/workflow.md",
                "usage/memory.md",
                "usage/conditional.md",
                "usage/faq.md",
            ],
        ]
    )
end

isinteractive() || main()
