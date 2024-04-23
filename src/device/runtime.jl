# CUDA-specific runtime libraries

import Base.Sys: WORD_SIZE


## GPU runtime library

# reset the runtime cache from global scope, so that any change triggers recompilation
GPUCompiler.reset_runtime()

# load or build the runtime for the most likely compilation jobs
function precompile_runtime()
    f = ()->return
    mi = methodinstance(typeof(f), Tuple{})

    caps = llvm_compat().cap
    ptx = maximum(llvm_compat().ptx)
    JuliaContext() do ctx
        for cap in caps, debuginfo in [false, true]
            # NOTE: this often runs when we don't have a functioning set-up,
            #       so we don't use `compiler_config` which requires NVML
            target = PTXCompilerTarget(; cap, ptx, debuginfo)
            params = CUDACompilerParams(; cap, ptx)
            config = CompilerConfig(target, params)
            job = CompilerJob(mi, config)
            GPUCompiler.load_runtime(job)
        end
    end
    return
end

struct KernelState
    exception_flag::Ptr{UInt8}
    random_seed::UInt32
end

@inline @generated kernel_state() = GPUCompiler.kernel_state_value(KernelState)

function signal_exception()
    ptr = kernel_state().exception_flag
    if ptr !== C_NULL
        unsafe_store!(ptr, 1)
        threadfence_system()
    else
        @cuprintf("""
            WARNING: could not signal exception status to the host, execution will continue.
                     Please file a bug.
            """)
    end
    exit()
    return
end

function report_exception(ex)
    @cuprintf("""
        ERROR: a %s was thrown during kernel execution.
        For stacktrace reporting, run Julia on debug level 2 (by passing -g2 to the executable).
        """, ex)
    return
end

function report_oom(sz)
    @cuprintf("ERROR: Out of dynamic GPU memory (trying to allocate %i bytes)\n", sz)
    return
end

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
