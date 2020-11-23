# CUDA-specific runtime libraries


## GPU runtime library

# reset the runtime cache from global scope, so that any change triggers recompilation
GPUCompiler.reset_runtime()

# load or build the runtime for the most likely compilation job given a compute capability
function precompile_runtime(caps=llvm_cap_support(LLVM.version()))
    dummy_source = FunctionSpec(()->return, Tuple{})
    params = CUDACompilerParams()
    JuliaContext() do ctx
        for cap in caps
            target = PTXCompilerTarget(; cap=cap)
            job = CompilerJob(target, dummy_source, params)
            GPUCompiler.load_runtime(job, ctx)
        end
    end
    return
end

@inline exception_flag() = ccall("extern julia_exception_flag", llvmcall, Ptr{Cvoid}, ())

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

function report_exception(ex)
    @cuprintf("""
        ERROR: a %s was thrown during kernel execution.
               Run Julia on debug level 2 for device stack traces.
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


## CUDA device library

function load_libdevice(cap, ctx)
    path = libdevice()
    parse(LLVM.Module, read(path), ctx)
end

function link_libdevice!(mod::LLVM.Module, cap::VersionNumber, undefined_fns)
    ctx = LLVM.context(mod)

    # only link if there's undefined __nv_ functions
    if !any(fn->startswith(fn, "__nv_"), undefined_fns)
        return
    end
    lib::LLVM.Module = load_libdevice(cap, ctx)

    # override libdevice's triple and datalayout to avoid warnings
    triple!(lib, triple(mod))
    datalayout!(lib, datalayout(mod))

    GPUCompiler.link_library!(mod, lib)

    ModulePassManager() do pm
        push!(metadata(mod), "nvvm-reflect-ftz",
              MDNode([ConstantInt(Int32(1), ctx)]))
        run!(pm, mod)
    end
end
