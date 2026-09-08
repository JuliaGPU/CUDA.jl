using Base.CoreLogging: Debug

@testset "driver log" begin
    if CUDA.driver_version() >= v"12.9"
        CUDA.enable_logging()
        try
            # the driver explains failures, including ones that never surface as an exception
            @test_logs (:debug, r"CUDA_ERROR_INVALID_VALUE") min_level=Debug match_mode=:any begin
                CUDACore.unchecked_cuMemAlloc_v2(Ref{CUDACore.CUdeviceptr}(), 0)
                CUDACore.flush_logs()
            end

            # messages are attributed to the package
            logs, _ = Test.collect_test_logs(min_level=Debug) do
                CUDACore.unchecked_cuMemAlloc_v2(Ref{CUDACore.CUdeviceptr}(), 0)
                CUDACore.flush_logs()
            end
            @test !isempty(logs)
            @test all(log -> log._module === CUDACore, logs)

            # exceptions still carry the driver's explanation, independent of forwarding
            err = try
                CUDACore.cuMemAlloc_v2(Ref{CUDACore.CUdeviceptr}(), 0)
                nothing
            catch err
                err
            end
            @test err isa CuError
            @test err.log[] !== nothing
            @test occursin("CUDA_ERROR_INVALID_VALUE", err.log[])
            @test occursin("Driver log:", sprint(showerror, err))

            # messages from other tasks are delivered to that task's logger
            logs, _ = Test.collect_test_logs(min_level=Debug) do
                fetch(Threads.@spawn begin
                    CUDACore.unchecked_cuMemAlloc_v2(Ref{CUDACore.CUdeviceptr}(), 0)
                    CUDACore.flush_logs()
                end)
            end
            @test any(log -> occursin("CUDA_ERROR_INVALID_VALUE", log.message), logs)

            CUDA.enable_logging(false)
            @test_logs min_level=Debug begin
                CUDACore.unchecked_cuMemAlloc_v2(Ref{CUDACore.CUdeviceptr}(), 0)
                CUDACore.flush_logs()
            end
        finally
            CUDA.enable_logging(false)
        end
    else
        @test_logs (:warn, r"requires CUDA 12.9") CUDA.enable_logging()
    end
end

@testset "JULIA_DEBUG=CUDA" begin
    for (debug, core, blas) in (("CUDA", true, true),
                               ("CUDACore", true, false),
                               ("cuBLAS", false, true),
                               ("", false, false),
                               ("CUDA,!cuBLAS", true, false),
                               ("all,!CUDA", false, false))
        withenv("JULIA_DEBUG" => debug) do
            @test CUDACore.isdebug() == core
            @test CUDACore.isdebug(CUDA.cuBLAS) == blas
            io = IOBuffer()
            Base.CoreLogging.with_logger(Base.CoreLogging.SimpleLogger(io)) do
                CUDACore.enqueue_log(CUDACore, Debug, "core message")
                CUDACore.enqueue_log(CUDA.cuBLAS, Debug, "BLAS message")
                CUDA.flush_logs()
            end
            output = String(take!(io))
            @test occursin("core message", output) == core
            @test occursin("BLAS message", output) == blas
            @test ENV["JULIA_DEBUG"] == debug
        end
    end
end

@testset "callback safety" begin
    # log callbacks must never throw
    CUDACore.init_logging()
    @test_logs (:error, r"Error in log callback.*boom"s) begin
        CUDACore.guarded_callback() do
            error("boom")
        end
        CUDACore.flush_logs()
    end
end

struct BlockingLibraryLogger <: Base.CoreLogging.AbstractLogger
    entered::Channel{Nothing}
    release::Channel{Nothing}
end
Base.CoreLogging.min_enabled_level(::BlockingLibraryLogger) = Debug
Base.CoreLogging.shouldlog(::BlockingLibraryLogger, args...) = true
function Base.CoreLogging.handle_message(logger::BlockingLibraryLogger, args...; kwargs...)
    put!(logger.entered, nothing)
    take!(logger.release)
end

struct FailingLibraryLogger <: Base.CoreLogging.AbstractLogger end
Base.CoreLogging.min_enabled_level(::FailingLibraryLogger) = Debug
Base.CoreLogging.shouldlog(::FailingLibraryLogger, args...) = true
Base.CoreLogging.catch_exceptions(::FailingLibraryLogger) = false
Base.CoreLogging.handle_message(::FailingLibraryLogger, args...; kwargs...) = error("logger failed")

@testset "log delivery" begin
    CUDACore.init_logging()
    CUDA.flush_logs()

    # Delivery keeps the originating logger even after with_logger returns.
    logger = Test.TestLogger(min_level=Debug)
    Base.CoreLogging.with_logger(logger) do
        CUDACore.enqueue_log(CUDACore, Debug, "captured")
    end
    CUDA.flush_logs()
    @test only(logger.logs).message == "captured"

    # A manual flush must wait for a background delivery that has already taken the queue.
    blocking = BlockingLibraryLogger(Channel{Nothing}(1), Channel{Nothing}(1))
    Base.CoreLogging.with_logger(blocking) do
        CUDACore.enqueue_log(CUDACore, Debug, "blocking")
    end
    take!(blocking.entered)
    flushing = @async CUDA.flush_logs()
    try
        yield()
        @test !istaskdone(flushing)
    finally
        put!(blocking.release, nothing)
        wait(flushing)
    end

    # A bad logger must not discard the rest of a batch or stop background delivery.
    redirect_stderr(devnull) do
        Base.CoreLogging.with_logger(FailingLibraryLogger()) do
            CUDACore.enqueue_log(CUDACore, Debug, "failure")
        end
        Base.CoreLogging.with_logger(logger) do
            CUDACore.enqueue_log(CUDACore, Debug, "after failure")
        end
        @test timedwait(() -> length(logger.logs) == 2, 10) == :ok
    end
    Base.CoreLogging.with_logger(logger) do
        CUDACore.enqueue_log(CUDACore, Debug, "next batch")
    end
    @test timedwait(() -> length(logger.logs) == 3, 10) == :ok
end
