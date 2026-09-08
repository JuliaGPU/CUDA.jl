using Base.CoreLogging: Debug
using Test: TestLogger, collect_test_logs

@testset "logging" begin
    cuBLAS.enable_logging()
    try
        A = CuArray(rand(Float32, 8, 8))

        # API calls are traced at debug level, as a single message per call
        logs, _ = collect_test_logs(min_level=Debug) do
            A * A
            CUDACore.flush_logs()
        end
        @test all(log -> log._module === cuBLAS, logs)
        if !Sys.iswindows()
            gemm = filter(log -> occursin(r"cublas[SD]?[Gg]emm", log.message), logs)
            @test !isempty(gemm)
            @test occursin("transa", first(gemm).message)
        end
        # cuBLASLt messages are forwarded too
        @test any(log -> occursin("cublasLt", log.message), logs)

        # Check severity mapping without depending on toolkit-specific performance hints.
        @test_logs (:info, "matmul: consider padding") begin
            ccall(@cfunction(cuBLAS.lt_log_message, Nothing, (Int32, Cstring, Cstring)),
                  Nothing, (Int32, Cstring, Cstring), 3, "matmul", "consider padding")
            CUDACore.flush_logs()
        end

        # cuBLASXt logs from worker threads, whose messages go to the global logger
        if !Sys.iswindows()
            logger = TestLogger(min_level=Debug)
            old_logger = Base.CoreLogging.global_logger(logger)
            try
                Base.CoreLogging.with_logger(logger) do
                    n = 256
                    C = zeros(Float32, n, n)
                    cuBLAS.xt_gemm!('N', 'N', 1f0, rand(Float32, n, n), rand(Float32, n, n), 0f0, C)
                    CUDACore.flush_logs()
                end
            finally
                Base.CoreLogging.global_logger(old_logger)
            end
            @test any(log -> occursin("cublasXtSgemm", log.message), logger.logs)
            @test any(log -> occursin("cublasSgemm_v2", log.message), logger.logs)
        end

        cuBLAS.enable_logging(false)
        @test_logs min_level=Debug begin
            A * A
            CUDACore.flush_logs()
        end
    finally
        cuBLAS.enable_logging(false)
    end
end

@testset "message assembly" begin
    CUDACore.init_logging()
    cuBLAS.flush_log_buffers()
    CUDA.flush_logs()
    @test_logs (:warn, "example\nα parameter\nTime: now\nProcess: test") begin
        callback = @cfunction(cuBLAS.log_message, Nothing, (Cstring,))
        # Some chunks have no newline; unmarked lines may start with Unicode.
        for line in ("W! example", "α parameter", "Time: now\nProcess: test\n")
            ccall(callback, Nothing, (Cstring,), line)
        end
        CUDA.flush_logs()
    end
end
