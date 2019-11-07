using Documenter, Literate
using CUDAapi, CUDAdrv, CUDAnative, CuArrays

@info "Building Literate.jl documentation"
function set_repo_root(content)
    if haskey(ENV, "GITLAB_CI")
        commit = ENV["CI_COMMIT_REF_NAME"]
        repo_root_url = "https://github.com/JuliaGPU/CUDA.jl/blob/$(commit)"
        return replace(content, "@__REPO_ROOT_URL__" => repo_root_url)
    else
        return content
    end
end
cd(joinpath(@__DIR__, "src")) do
    withenv("TRAVIS_REPO_SLUG" => "JuliaGPU/CUDA.jl") do
        Literate.markdown("tutorials/introduction.jl", "tutorials"; postprocess=set_repo_root)
    end
end

@info "Generating Documenter.jl site"
makedocs(
    sitename = "CUDA.jl",
    authors = "Tim Besard",
    repo = "https://github.com/JuliaGPU/CUDA.jl/blob/{commit}{path}#{line}",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    doctest = false,
    pages = Any[
        "Home" => "index.md",
        "Tutorials" => Any[
            "tutorials/introduction.md",
        ],
        "Usage" => Any[
            "usage/overview.md",
            "usage/conditional.md",
            "usage/faq.md",
        ],
        "Development" => Any[
            "development/workflow.md",
        ],
      ]
)
