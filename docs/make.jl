using Documenter, CUDAdrv

const src = "https://github.com/JuliaGPU/CUDAdrv.jl"
const dst = "https://juliagpu.gitlab.io/CUDAdrv.jl/"

function main()
    makedocs(
        sitename = "CUDAdrv.jl",
        authors = "Tim Besard",
        repo = "$src/blob/{commit}{path}#{line}",
        format = Documenter.HTML(
            # Use clean URLs on CI
            prettyurls = get(ENV, "CI", nothing) == "true",
            canonical = dst,
            assets = ["assets/favicon.ico"],
            analytics = "UA-154489943-5",
        ),
        doctest = false,
        pages = Any[
            "Home"    => "index.md",
            "APIs" => [
                "driver.md",
            ]
        ]
    )
end

isinteractive() || main()
