using Documenter, CUDAnative

const src = "https://github.com/JuliaGPU/CUDAnative.jl"
const dst = "https://juliagpu.gitlab.io/CUDAnative.jl/"

function main()
    makedocs(
        sitename = "CUDAnative.jl",
        authors = "Tim Besard",
        repo = "$src/blob/{commit}{path}#{line}",
        format = Documenter.HTML(
            # Use clean URLs on CI
            prettyurls = get(ENV, "CI", nothing) == "true",
            canonical = dst,
            assets = ["assets/favicon.ico"],
            analytics = "UA-154489943-4",
        ),
        doctest = false,
        pages = Any[
            "Home"    => "index.md",
            "Host" => [
                "host/initialization.md",
                "host/execution.md",
                "host/reflection.md",
            ],
            "Device" => [
                "device/cuda.md",
                "device/wmma.md",
                "device/array.md"
            ]
        ]
    )
end

isinteractive() || main()
