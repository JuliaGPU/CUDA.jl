using CUTENSOR, CUDA, Test

@testset "CUTENSOR" begin

# include all tests
for entry in readdir(@__DIR__)
    endswith(entry, ".jl") || continue
    entry in ["runtests.jl"] && continue

    # generate a testset
    name = splitext(entry)[1]
    @eval begin
        @testset $name begin
            include($entry)
        end
    end
end

end
