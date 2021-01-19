using Documenter, Literate
using CUDA

const src = "https://github.com/JuliaGPU/CUDA.jl"
const dst = "https://juliagpu.github.io/CUDA.jl/stable/"

function main()
    ci = get(ENV, "CI", "") == "true"

    @info "Building Literate.jl documentation"
    cd(@__DIR__) do
        for filname in ["introduction.jl", "sum.jl"]
            Literate.markdown(
                joinpath("src", "tutorials", filname),
                joinpath("src", "tutorials"),
                repo_root_url="$src/blob/master/docs")
        end
    end

    @info "Generating Documenter.jl site"
    DocMeta.setdocmeta!(CUDA, :DocTestSetup, :(using CUDA); recursive=true)
    makedocs(
        sitename = "CUDA.jl",
        authors = "Tim Besard",
        repo = "$src/blob/{commit}{path}#{line}",
        format = Documenter.HTML(
            # Use clean URLs on CI
            prettyurls = ci,
            canonical = dst,
            assets = ["assets/favicon.ico"],
            analytics = "UA-154489943-2",
        ),
        doctest = true,
        #strict = true,
        modules = [CUDA],
        pages = Any[
            "Home" => "index.md",
            "Tutorials" => Any[
                "tutorials/introduction.md",
                "tutorials/sum.md",
            ],
            "Installation" => Any[
                "installation/overview.md",
                "installation/conditional.md",
                "installation/troubleshooting.md",
            ],
            "Usage" => Any[
                "usage/overview.md",
                "usage/workflow.md",
                "usage/array.md",
                "usage/memory.md",
                "usage/multigpu.md",
            ],
            "Development" => Any[
                "development/profiling.md",
                "development/troubleshooting.md",
            ],
            "API reference" => Any[
                "api/essentials.md",
                "api/compiler.md",
                "api/kernel.md",
                "api/array.md",
            ],
            "Library reference" => Any[
                "lib/driver.md",
            ],
            "FAQ" => "faq.md",
        ]
    )

    if ci
        @info "Deploying to GitHub"
        deploydocs(
            repo = "github.com/JuliaGPU/CUDA.jl.git",
            push_preview = true
        )
    end
end

isinteractive() || main()
