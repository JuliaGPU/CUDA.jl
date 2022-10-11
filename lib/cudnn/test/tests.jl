module Tests

using CUDNN, CUDA, ReTest

for entry in readdir(@__DIR__)
    endswith(entry, ".jl") || continue
    entry in ["runtests.jl", "tests.jl"] && continue

    # XXX: disabled due to sporadic CI issue (JuliaGPU/CUDA.jl#/725)
    entry == "convolution.jl" && continue

    # generate a testset
    name = splitext(entry)[1]
    @eval begin
        @testset $name begin
            include($entry)
        end
    end
end

end
Tests.runtests()
