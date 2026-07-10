using cuTENSOR
using CUDACore

if capability(device()) >= v"8.0"
    @testset "kernel cache" begin
        mktempdir() do dir
            cd(dir) do
                supported = try
                    cuTENSOR.write_cache!("kernelCache.bin")
                    true
                catch err
                    # JIT compilation may not support the current device
                    (isa(err, CUTENSORError) &&
                     err.code == cuTENSOR.CUTENSOR_STATUS_NOT_SUPPORTED) || rethrow()
                    false
                end
                if supported
                    @test isfile("kernelCache.bin")
                    cuTENSOR.read_cache!("kernelCache.bin")
                end
            end
        end
    end
end
