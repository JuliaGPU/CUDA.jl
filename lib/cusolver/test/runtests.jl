include("setup.jl")

@testset verbose=true "cuSOLVER" begin
    for (root, _, files) in walkdir(@__DIR__)
        for file in sort(files)
            endswith(file, ".jl") || continue
            file in ("setup.jl", "runtests.jl") && continue
            include(joinpath(root, file))
        end
    end
end
