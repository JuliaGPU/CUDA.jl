# CUDA-specific interfaces and extensions to the GPU runtime library

# TODO: wipe the cached runtime library during precompilation

# TODO: compile should be per target

## exception handling

function report_exception(ex)
    @cuprintf("""
        ERROR: a %s was thrown during kernel execution.
               Run Julia on debug level 2 for device stack traces.
        """, ex)
    return
end

report_oom(sz) = @cuprintf("ERROR: Out of dynamic GPU memory (trying to allocate %i bytes)\n", sz)

function report_exception_name(ex)
    @cuprintf("""
        ERROR: a %s was thrown during kernel execution.
        Stacktrace:
        """, ex)
    return
end

function report_exception_frame(idx, func, file, line)
    @cuprintf(" [%i] %s at %s:%i\n", idx, func, file, line)
    return
end

@inline exception_flag() =
    ccall("extern cudanativeExceptionFlag", llvmcall, Ptr{Cvoid}, ())

function signal_exception()
    ptr = exception_flag()
    if ptr !== C_NULL
        unsafe_store!(convert(Ptr{Int}, ptr), 1)
        threadfence_system()
    else
        @cuprintf("""
            WARNING: could not signal exception status to the host, execution will continue.
                     Please file a bug.
            """)
    end
    return
end
