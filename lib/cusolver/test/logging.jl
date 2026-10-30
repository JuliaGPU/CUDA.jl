using Base.CoreLogging: Debug, Info, Error

@testset "logging" begin
    if !cuSOLVER.has_logging_api()
        # the shared library does not export the logging API (only the static one defines it)
        isempty(cuSOLVER.log_file[]) && @test_logs (:warn, r"does not export the logging callback API") cuSOLVER.enable_logging()
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

    @testset "parsing the log file" begin
        prefix = "[2026-10-30 05:01:05][cusolverDn][1234]"
        @test cuSOLVER.parse_log_line("$prefix[Api][cusolverDnCreate] handle=0x1") ==
              (Debug, "cusolverDnCreate: handle=0x1")
        @test cuSOLVER.parse_log_line("$prefix[Api][cusolverDnCreate] ") ==
              (Debug, "cusolverDnCreate(...)")
        @test cuSOLVER.parse_log_line("$prefix[Hint][cusolverDnCreate] slow") ==
              (Info, "cusolverDnCreate: slow")
        @test cuSOLVER.parse_log_line("$prefix[Error][cusolverDnCreate] bad") ==
              (Error, "cusolverDnCreate: bad")
        @test cuSOLVER.parse_log_line("not a log line") == (Debug, "not a log line")
    end

    # the library only reads its environment once, so this needs a fresh process
    cuSOLVER.has_logging_api() || @testset "JULIA_DEBUG=$debug" for debug in ("cuSOLVER", "CUDA")
        script = """
            using CUDACore, cuSOLVER, LinearAlgebra, Logging
            A = CuArray(rand(Float32, 8, 8))
            function logged()
                io = IOBuffer()
                with_logger(SimpleLogger(io, Logging.Debug)) do
                    lu(A)
                    CUDACore.flush_logs()
                end
                return String(take!(io))
            end
            @assert occursin("cusolverDnSgetrf", logged())
            @assert !haskey(ENV, "CUSOLVERDN_LOG_LEVEL") && !haskey(ENV, "CUSOLVERDN_LOG_FILE")
            cuSOLVER.enable_logging(false)
            @assert !occursin("cusolverDn", logged())
            cuSOLVER.enable_logging(true)
            @assert occursin("cusolverDnSgetrf", logged())
            """
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`
        @test success(pipeline(addenv(cmd, "JULIA_DEBUG" => debug,
                                      "CUSOLVERDN_LOG_LEVEL" => nothing,
                                      "CUSOLVERDN_LOG_FILE" => nothing);
                               stdout, stderr))
    end
end
