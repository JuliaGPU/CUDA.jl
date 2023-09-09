using Test

using CUDA
@info "CUDA information:\n" * sprint(io->CUDA.versioninfo(io))

using cuDNN
@test cuDNN.has_cudnn()
@info "cuDNN version: $(cuDNN.version()) (built for CUDA $(cuDNN.cuda_version()))"

@testset "cuDNN" begin

# include all tests
for entry in readdir(@__DIR__)
    endswith(entry, ".jl") || continue
    entry in ["runtests.jl"] && continue

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
