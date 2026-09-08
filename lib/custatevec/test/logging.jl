using Base.CoreLogging: Debug

@testset "logging" begin
    cuStateVec.enable_logging()
    try
        sv = CuStateVec(ComplexF32, 2)
        X = ComplexF32[0 1; 1 0]

        @test_logs (:debug, r"custatevecApplyMatrix") min_level=Debug match_mode=:any begin
            applyMatrix!(sv, X, false, Int32[0], Int32[])
            synchronize()
            CUDACore.flush_logs()
        end

        cuStateVec.enable_logging(false)
        @test_logs min_level=Debug begin
            applyMatrix!(sv, X, false, Int32[0], Int32[])
            synchronize()
            CUDACore.flush_logs()
        end
    finally
        cuStateVec.enable_logging(false)
    end
end
