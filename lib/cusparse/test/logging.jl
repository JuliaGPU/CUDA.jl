using Base.CoreLogging: Debug

@testset "logging" begin
    if !cuSPARSE.has_logging_api()
        # the logging callback API requires CUDA 11 or later
        @test_logs (:warn, r"does not provide the logging callback API") cuSPARSE.enable_logging()
    else
        cuSPARSE.enable_logging()
        try
            A = CuSparseMatrixCSR(sprand(Float32, 16, 16, 0.5))
            x = CuArray(rand(Float32, 16))

            # API calls and kernel launches are traced at debug level
            logs, _ = Test.collect_test_logs(min_level=Debug) do
                A * x
                CUDACore.flush_logs()
            end
            @test all(log -> log._module === cuSPARSE, logs)
            @test any(log -> occursin(r"^cusparseSpMV", log.message), logs)

            cuSPARSE.enable_logging(false)
            @test_logs min_level=Debug begin
                A * x
                CUDACore.flush_logs()
            end
        finally
            cuSPARSE.enable_logging(false)
        end
    end
end
