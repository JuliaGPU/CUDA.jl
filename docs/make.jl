using Documenter, Literate
using CuArrays

const src = "https://github.com/JuliaGPU/CuArrays.jl"
const dst = "https://juliagpu.gitlab.io/CuArrays.jl/"

function main()
    makedocs(
        sitename = "CuArrays.jl",
        authors = "Tim Besard",
        repo = "$src/blob/{commit}{path}#{line}",
        format = Documenter.HTML(
            # Use clean URLs on CI
            prettyurls = get(ENV, "CI", nothing) == "true",
            canonical = dst,
            assets = ["assets/favicon.ico"],
            analytics = "UA-154489943-3",
        ),
        doctest = false,
        pages = Any[
            "Home" => "index.md",
            "APIs" => Any[
                "memory.md",
            ],
            "Libraries" => Any[
                "lib/blas.md",
                "lib/rand.md",
                "lib/fft.md",
                "lib/solver.md",
                "lib/sparse.md",
                "lib/dnn.md",
                "lib/tensor.md",
            ],
        ]
    )
end

isinteractive() || main()
