using Weave

using Pkg
if haskey(ENV, "GITLAB_CI")
    Pkg.add([PackageSpec(name = x; rev = "master")
             for x in ["CUDAapi", "GPUArrays", "CUDAnative", "NNlib", "CUDAdrv"]])
end

cd(joinpath(@__DIR__, "src")) do
    rm("../build"; force=true, recursive=true)

    # intro tutorial
    weave("intro.jl", out_path="../build", doctype="md2html")
    cp("intro1.png", "../build/intro1.png")
end
