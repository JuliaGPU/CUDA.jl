include("setup.jl")
@test cuDNN.functional()

@testset verbose=true "cuDNN" begin

# include all tests
for entry in readdir(@__DIR__)
    endswith(entry, ".jl") || continue
    entry in ["runtests.jl", "setup.jl"] && continue

    # generate a testset
    name = splitext(entry)[1]
    @eval begin
        @testset $name begin
            include($entry)
        end
    end
end

end
