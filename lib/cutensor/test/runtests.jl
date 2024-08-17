using Test

using CUDA
@info "CUDA information:\n" * sprint(io->CUDA.versioninfo(io))

using cuTENSOR
@test cuTENSOR.has_cutensor()
@info "cuTENSOR version: $(cuTENSOR.version()) (built for CUDA $(cuTENSOR.cuda_version()))"

@testset "cuTENSOR" begin

include("base.jl")

include("elementwise_binary.jl")
include("elementwise_trinary.jl")
include("permutations.jl")
include("contractions.jl")
include("reductions.jl")

# we should have some kernels in the cache after this
if CUDA.runtime_version() >= v"11.8" && capability(device()) >= v"8.0"
@testset "kernel cache" begin
    mktempdir() do dir
    cd(dir) do
        cuTENSOR.write_cache!("kernelCache.bin")
        @test isfile("kernelCache.bin")
        cuTENSOR.read_cache!("kernelCache.bin")
    end
    end
end
end

end
