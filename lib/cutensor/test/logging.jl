using Base.CoreLogging: Debug
using cuTENSOR: permute!

@testset "logging" begin
    cuTENSOR.enable_logging()
    try
        A = CuArray(rand(Float32, 4, 8))
        C = similar(A, 8, 4)

        @test_logs (:debug, r"cutensorPermute") min_level=Debug match_mode=:any begin
            permute!(1f0, A, ['a', 'b'], cuTENSOR.OP_IDENTITY, C, ['b', 'a'])
            synchronize()
            CUDACore.flush_logs()
        end

        cuTENSOR.enable_logging(false)
        @test_logs min_level=Debug begin
            permute!(1f0, A, ['a', 'b'], cuTENSOR.OP_IDENTITY, C, ['b', 'a'])
            synchronize()
            CUDACore.flush_logs()
        end
    finally
        cuTENSOR.enable_logging(false)
    end
end
