# CUDA-specific runtime libraries

import Base.Sys: WORD_SIZE


## GPU runtime library

# reset the runtime cache from global scope, so that any change triggers recompilation
GPUCompiler.reset_runtime()

# load or build the runtime for the most likely compilation job given a compute capability
function precompile_runtime(caps=CUDA.llvm_compat(LLVM.version()).cap)
    dummy_source = FunctionSpec(()->return, Tuple{})
    params = CUDACompilerParams()
    Context() do ctx
        for cap in caps
            # NOTE: this often runs when we don't have a functioning set-up,
            #       so we don't use CUDACompilerTarget(...) which requires NVML
            target = PTXCompilerTarget(; cap=cap)
            job = CompilerJob(target, dummy_source, params)
            GPUCompiler.load_runtime(job; ctx)
        end
    end
    return
end

struct KernelState
    exception_flag::LLVMPtr{Int8, AS.Global}

    hostcall_pointers::LLVMPtr{UInt32, AS.Global}
    hostcalls::LLVMPtr{Hostcall, AS.Global}
end

@inline @generated kernel_state() = GPUCompiler.kernel_state_value(KernelState)

# exception handling

exception_flag() = kernel_state().exception_flag

function signal_exception()
    unsafe_store!(exception_flag(), 1)
    threadfence_system()
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

# hostcall

hostcall_pointers() = CuDeviceArray(2, kernel_state().hostcall_pointers)

hostcalls() = CuDeviceArray(HOSTCALL_POOL_SIZE, kernel_state().hostcalls)

# generate accessors for individual fields
for i in 1:fieldcount(Hostcall)
    local typ = fieldtype(Hostcall, i)
    local name = fieldname(Hostcall, i)
    local offset = fieldoffset(Hostcall, i)

    local align = Base.datatype_alignment(typ)

    ptr = Symbol("hostcall_$(name)_ptr")
    getter = Symbol("hostcall_$(name)")
    setter = Symbol("hostcall_$(name)!")
    @eval begin
        $(ptr)(i=1) =
            reinterpret(LLVMPtr{$typ,AS.Global}, pointer(hostcalls(), i)) + $offset
        $(getter)(i=1) = unsafe_load($(ptr)(i), 1, Val($align))
        $(setter)(x::$typ, i=1) = unsafe_store!($(ptr)(i), x, 1, Val($align))
    end
end


## CUDA device library

function load_libdevice(cap; ctx)
    path = libdevice()
    parse(LLVM.Module, read(path); ctx)
end

function link_libdevice!(mod::LLVM.Module, cap::VersionNumber, undefined_fns)
    ctx = LLVM.context(mod)

    # only link if there's undefined __nv_ functions
    if !any(fn->startswith(fn, "__nv_"), undefined_fns)
        return
    end
    lib::LLVM.Module = load_libdevice(cap; ctx)

    # override libdevice's triple and datalayout to avoid warnings
    triple!(lib, triple(mod))
    datalayout!(lib, datalayout(mod))

    GPUCompiler.link_library!(mod, lib)

    ModulePassManager() do pm
        push!(metadata(mod)["nvvm-reflect-ftz"],
              MDNode([ConstantInt(Int32(1); ctx)]; ctx))
        run!(pm, mod)
    end
end
