using Documenter, Literate
using CUDA

const src = "https://github.com/JuliaGPU/CUDA.jl"
const dst = "https://cuda.juliagpu.org/stable/"

function main()
    ci = get(ENV, "CI", "") == "true"

    @info "Building Literate.jl documentation"
    cd(@__DIR__) do
        Literate.markdown("src/tutorials/introduction.jl", "src/tutorials";
                          repo_root_url="$src/blob/master/docs")
        Literate.markdown("src/tutorials/custom_structs.jl", "src/tutorials";
                          repo_root_url="$src/blob/master/docs")
        Literate.markdown("src/tutorials/performance.jl", "src/tutorials";
                          repo_root_url="$src/blob/master/docs")
    end
    println()

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
        warnonly = [:missing_docs],
        modules = [CUDA],
        pages = Any[
            "Home" => "index.md",
            "Tutorials" => Any[
                "tutorials/introduction.md",
                "tutorials/custom_structs.md",
                "tutorials/performance.md"
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
                "usage/multitasking.md",
                "usage/multigpu.md",
            ],
            "Development" => Any[
                "development/profiling.md",
                "development/kernel.md",
                "development/troubleshooting.md",
                "development/debugging.md",
            ],
            "Hacking" => Any[
                "hacking/exposing_new_intrinsics.md",
            ],
            "API reference" => Any[
                "api/essentials.md",
                "api/array.md",
                "api/kernel.md",
                "api/compiler.md",
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
