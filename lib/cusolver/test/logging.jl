using Base.CoreLogging: Debug

@testset "logging" begin
    if !cuSOLVER.has_logging_api()
        # the shared library does not export the logging API (only the static one defines it)
        @test_logs (:warn, r"does not export the logging callback API") cuSOLVER.enable_logging()
    else
        cuSOLVER.enable_logging()
        try
            A = CuArray(rand(Float32, 8, 8))
            @test_logs (:debug, r"cusolverDn\w*getrf") min_level=Debug match_mode=:any begin
                lu(A)
                CUDACore.flush_logs()
            end

            cuSOLVER.enable_logging(false)
            @test_logs min_level=Debug begin
                lu(A)
                CUDACore.flush_logs()
            end
        finally
            cuSOLVER.enable_logging(false)
        end
    end
end
