using Documenter, Literate
using CUDA

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
        doctest = true,
        pages = Any[
            "Home" => "index.md",
            "Tutorials" => Any[
                "tutorials/introduction.md",
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
end

isinteractive() || main()
