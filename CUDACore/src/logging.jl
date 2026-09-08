# logging functionality: forwarding messages from the driver and libraries to Julia

using Base.CoreLogging: LogLevel, Debug, Info, Warn, Error, @logmsg

@public enable_logging, flush_logs


## debug helpers

# A shared group makes JULIA_DEBUG=CUDA cover the stack while preserving module filters.
isdebug(mod=CUDACore; group=:CUDA) =
    Base.CoreLogging.current_logger_for_env(Debug, group, mod) !== nothing


## message queue
#
# Callbacks can run on NVIDIA worker threads while the calling thread waits in the library.
# They must not yield, perform I/O, re-enter the library, or throw through its C frames.
# Queue messages under a spin lock; a Julia task delivers them to the logger.

struct LibraryLogMessage
    mod::Module
    level::LogLevel
    message::String
    # the logging state of the task that triggered the message, so that e.g. `with_logger`
    # around a library call captures its messages (for foreign threads: the global logger)
    logstate::Base.CoreLogging.LogState
end

const log_queue = LibraryLogMessage[]
const log_queue_lock = Threads.SpinLock()    # a spin lock, so that taking it can't yield
const log_condition = Ref{Base.AsyncCondition}()
const log_condition_lock = ReentrantLock()
const log_flush_lock = ReentrantLock()

# set up the queue. must be called before registering any callback with the driver or a
# library, and creates a task, so cannot be called during precompilation.
function init_logging()
    Base.@lock log_condition_lock begin
        if !isassigned(log_condition) || !isopen(log_condition[])
            log_condition[] = Base.AsyncCondition() do _
                flush_logs()
            end
            # messages that are still queued at exit would otherwise be lost
            atexit(flush_logs)
        end
    end
    return
end

# queue a message. safe to call from a log callback.
function enqueue_log(mod::Module, level::LogLevel, message::AbstractString)
    msg = LibraryLogMessage(mod, level, String(message),
                            Base.CoreLogging.current_logstate())
    Base.@lock log_queue_lock push!(log_queue, msg)
    # NOTE: uv_async_send is thread-safe, but coalesces wake-ups, so the handler drains the
    #       entire queue rather than processing a single message.
    if isassigned(log_condition)
        ccall(:uv_async_send, Cint, (Ptr{Cvoid},), log_condition[].handle)
    end
    return
end

"""
    CUDA.flush_logs()

Deliver queued driver and library messages to Julia's logging system, waiting for any
ongoing delivery to finish. Call this before inspecting captured messages or closing a
logger's output stream. This does not synchronize GPU work or enable logging.
"""
function flush_logs()
    # Only delivery takes this lock; callbacks must remain able to enqueue while a logger
    # yields or calls a CUDA library itself.
    Base.@lock log_flush_lock begin
        messages = Base.@lock log_queue_lock begin
            isempty(log_queue) && return
            queued = copy(log_queue)
            empty!(log_queue)
            queued
        end
        for msg in messages
            try
                Base.CoreLogging.with_logstate(() -> emit_log(msg), msg.logstate)
            catch err
                # A closed stream or failing custom logger must not kill the delivery task
                # and leave all subsequent messages accumulating in the queue.
                try
                    Base.display_error(stderr, err, catch_backtrace())
                catch
                end
            end
        end
    end
    return
end

function emit_log(msg::LibraryLogMessage)
    @logmsg msg.level msg.message _module=msg.mod _group=:CUDA _file=nothing _line=nothing
    return
end

# run the body of a log callback, guarding against things that are invalid in a callback:
# - throwing, which would unwind through the driver's or library's frames;
# - running finalizers, which could call back into the library that invoked the callback,
#   while it is still holding internal locks.
function guarded_callback(f)
    # don't use `GC.enable_finalizers(true)` to re-enable, as that immediately runs any
    # pending finalizers; they will be run at the next safe opportunity instead.
    ccall(:jl_gc_disable_finalizers_internal, Cvoid, ())
    try
        f()
    catch err
        # report through the queue, since regular error reporting involves I/O
        try
            enqueue_log(CUDACore, Error,
                        "Error in log callback: " * sprint(showerror, err, catch_backtrace()))
        catch
        end
    finally
        ccall(:jl_gc_enable_finalizers_internal, Cvoid, ())
    end
    return
end


## library messages
#
# cuBLASLt, cuSPARSE, cuSOLVER, cuTENSOR and the cuQuantum libraries share a logging design,
# with a callback that receives a numeric level, the name of the API function, and a message:
#   1: errors, 2: trace (kernel launches), 3: performance hints, 4: info, 5: API trace

function library_log_callback(mod::Module, level::Integer, function_name::Cstring,
                              message::Cstring)
    guarded_callback() do
        level = if level <= 1
            Error
        elseif level == 3
            Info
        else
            Debug
        end

        function_name = unsafe_string(function_name)
        message = unsafe_string(message)
        output = if isempty(message)
            "$function_name(...)"
        else
            "$function_name: $message"
        end

        enqueue_log(mod, level, output)
    end
    return
end

# libraries also write to stdout when logging is enabled, unless a log file is set
const devnull_path = Sys.iswindows() ? "NUL" : "/dev/null"


## driver messages
#
# CuError captures the driver's ring buffer independently (see error.jl). Forward failures
# at Debug because they include handled errors as well as explanations already in CuError.

const driver_log_handle = Ref{CUlogsCallbackHandle}(C_NULL)
const driver_log_callback_lock = ReentrantLock()

function driver_log_callback(data::Ptr{Cvoid}, level::CUlogLevel, message::Ptr{UInt8},
                             length::Csize_t)
    guarded_callback() do
        jl_level = level == CU_LOG_LEVEL_WARNING ? Warn : Debug
        enqueue_log(CUDACore, jl_level, unsafe_string(message, length))
    end
    return
end

"""
    CUDA.enable_logging(enable::Bool=true)

Forward the CUDA driver's log messages to Julia's logging system. Failure explanations
are reported at `Debug` level and warnings at `Warn` level. This includes failures handled
internally by CUDA.jl or a library. Requires a driver supporting CUDA 12.9 or newer.

Starting Julia with `JULIA_DEBUG=CUDA` enables this automatically and shows debug messages
from all CUDA.jl packages. Calling this function only toggles driver log forwarding; it
does not change Julia's log level or enable library logging. Use [`flush_logs`](@ref) to
finish delivery before inspecting captured messages.

Driver explanations in `CuError` exceptions are available independently of this setting.
For crash diagnostics, set `CUDA_LOG_FILE` before starting Julia to write logs directly.
"""
function enable_logging(enable::Bool=true)
    if driver_version() < v"12.9"
        enable && @warn "Forwarding the driver log requires CUDA 12.9 or newer (found CUDA $(driver_version()))" maxlog=1
        return
    end

    Base.@lock driver_log_callback_lock begin
        if enable
            init_logging()
            if driver_log_handle[] == C_NULL
                callback = @cfunction(driver_log_callback, Nothing,
                                      (Ptr{Cvoid}, CUlogLevel, Ptr{UInt8}, Csize_t))
                cuLogsRegisterCallback(callback, C_NULL, driver_log_handle)
            end
        elseif driver_log_handle[] != C_NULL
            cuLogsUnregisterCallback(driver_log_handle[])
            driver_log_handle[] = C_NULL
        end
    end
    return
end
