# CUDA-specific runtime libraries

import Base.Sys: WORD_SIZE


## GPU runtime library

# reset the runtime cache from global scope, so that any change triggers recompilation
GPUCompiler.reset_runtime()

# load or build the runtime for the most likely compilation job given a compute capability
function precompile_runtime(caps=llvm_cap_support(LLVM.version()))
    dummy_source = FunctionSpec(()->return, Tuple{})
    params = CUDACompilerParams()
    JuliaContext() do ctx
        for cap in caps
            # NOTE: this often runs when we don't have a functioning set-up,
            #       so we don't use CUDACompilerTarget(...) which requires NVML
            target = PTXCompilerTarget(; cap=cap)
            job = CompilerJob(target, dummy_source, params)
            GPUCompiler.load_runtime(job, ctx)
        end
    end
    return
end

@eval @inline exception_flag() =
    Base.llvmcall(
        $("""@exception_flag = weak externally_initialized global i$(WORD_SIZE) 0
             define i64 @entry() #0 {
                 %ptr = load i$(WORD_SIZE), i$(WORD_SIZE)* @exception_flag, align 8
                 ret i$(WORD_SIZE) %ptr
             }
             attributes #0 = { alwaysinline }
          """, "entry"), Ptr{Cvoid}, Tuple{})

@inline cpucall_area() = ccall("extern julia_cpucall_area", llvmcall, Ptr{Cvoid}, ())

function acquire_lock(ptr::Core.LLVMPtr{Int64,AS.Global}, expected::Int64, value::Int64, id)
    try_count = 0
    ## TRY CAPTURE LOCK
    while atomic_cas!(ptr, expected, value) != expected && try_count < 500000
        try_count += 1
    end

    if try_count == 500000
        @cuprintln("Trycount $id maxed out")
    end
end

const IDLE = 0
const CPU_DONE = 1
const LOADING = 2
const CPU_CALL = 3
const CPU_HANDLING = 4

function call_syscall(load_f::Function, syscall::Int64, load_ret_f::Function)
    flag = cpucall_area()

    llvmptr    = reinterpret(Core.LLVMPtr{Int64,AS.Global}, flag)
    llvmptr_u8 = reinterpret(Core.LLVMPtr{UInt8,AS.Global}, flag)

    ## STORE ARGS
    acquire_lock(llvmptr, IDLE, LOADING, 1)
    load_f(llvmptr_u8 + 16)
    old = atomic_xchg!(llvmptr + 8, syscall)

    ## Notify CPU of syscall 'syscall'
    acquire_lock(llvmptr, LOADING, CPU_CALL, 2)

    try_count = 0
    while unsafe_load(llvmptr) != CPU_DONE && try_count < 500000
        try_count += 1
        threadfence_system()
    end
    if try_count == 500000
        @cuprintln("Special try_count maxed out")
    end

    ## GET RETURN ARGS
    acquire_lock(llvmptr, CPU_DONE, LOADING, 3)

    ret = load_ret_f(llvmptr_u8 + 16)

    ## Reset for next GPU syscall
    acquire_lock(llvmptr, LOADING, IDLE, 4)

    return ret
end

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
