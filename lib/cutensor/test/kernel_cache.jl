using cuTENSOR
using CUDACore

if CUDACore.runtime_version() >= v"11.8" && capability(device()) >= v"8.0"
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
