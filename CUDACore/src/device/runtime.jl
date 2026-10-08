# GPU runtime library

@public precompile_runtime

import Base.Sys: WORD_SIZE

# load or build the runtime for the most likely compilation jobs
function precompile_runtime()
    f = ()->return
    mi = methodinstance(typeof(f), Tuple{})

    # `.cap` is now keyed by `SMVersion` and includes variants; runtime caches are
    # feature_set-agnostic, so we only warm the baseline entries.
    sms = filter(sm -> sm.feature_set === :baseline, llvm_compat().sm)
    ptx = maximum(llvm_compat().ptx)
    JuliaContext() do ctx
        for sm in sms, debuginfo in [false, true]
            # NOTE: this often runs when we don't have a functioning set-up,
            #       so we don't use `compiler_config` which requires NVML
            target = PTXCompilerTarget(; cap=base_version(sm), ptx, debuginfo)
            params = CUDACompilerParams(; sm, ptx)
            config = CompilerConfig(target, params)
            job = CompilerJob(mi, config)
            GPUCompiler.load_runtime(job)
        end
    end
    return
end


## exception handling

# TODO: overloads in quirks.jl still do their own printing. maybe have them set an exception
#       reason in the info structure/

struct ExceptionInfo_st
    # whether an exception has been encountered (0 -> 1)
    status::Int32

    # whether an exception is being reported (see `lock_output!`)
    output_lock::Int32

    # who is reporting the exception
    thread::@NamedTuple{x::Int32,y::Int32,z::Int32}
    blockIdx::@NamedTuple{x::Int32,y::Int32,z::Int32}

    # any additional information
    subtype::Ptr{UInt8}
    reason::Ptr{UInt8}

    ExceptionInfo_st() = new(0, 0,
                             (; x=Int32(0), y=Int32(0), z=Int32(0)),
                             (; x=Int32(0), y=Int32(0), z=Int32(0)),
                             C_NULL, C_NULL)
end

# to simplify use of this struct, which is passed by-reference, use property overloading
const ExceptionInfo = Ptr{ExceptionInfo_st}
@inline function Base.getproperty(info::ExceptionInfo, sym::Symbol)
    if sym === :status
        unsafe_load(convert(Ptr{Int32}, info))
    elseif sym === :status_ptr
        reinterpret(LLVMPtr{Int32,AS.Generic}, info)
    elseif sym === :output_lock
        unsafe_load(convert(Ptr{Int32}, info + 4))
    elseif sym === :output_lock_ptr
        reinterpret(LLVMPtr{Int32,AS.Generic}, info + 4)
    elseif sym === :threadIdx
        unsafe_load(convert(Ptr{@NamedTuple{x::Int32,y::Int32,z::Int32}}, info + 8))
    elseif sym === :blockIdx
        unsafe_load(convert(Ptr{@NamedTuple{x::Int32,y::Int32,z::Int32}}, info + 20))
    elseif sym === :subtype
        unsafe_load(convert(Ptr{Ptr{UInt8}}, info + 32))
    elseif sym === :reason
        unsafe_load(convert(Ptr{Ptr{UInt8}}, info + 40))
    else
        getfield(info, sym)
    end
end
@inline function Base.setproperty!(info::ExceptionInfo, sym::Symbol, value)
    if sym === :status
        unsafe_store!(convert(Ptr{Int32}, info), value)
    elseif sym === :output_lock
        unsafe_store!(convert(Ptr{Int32}, info + 4), value)
    elseif sym === :threadIdx
        unsafe_store!(convert(Ptr{@NamedTuple{x::Int32,y::Int32,z::Int32}}, info + 8), value)
    elseif sym === :blockIdx
        unsafe_store!(convert(Ptr{@NamedTuple{x::Int32,y::Int32,z::Int32}}, info + 20), value)
    elseif sym === :subtype
        unsafe_store!(convert(Ptr{Ptr{UInt8}}, info + 32), value)
    elseif sym === :reason
        unsafe_store!(convert(Ptr{Ptr{UInt8}}, info + 40), value)
    else
        setfield!(info, sym, value)
    end
end

# a helper macro to generate global string pointers for storing additional details.
# this is used in quirk methods that replace exceptions from Base.
macro strptr(str::String)
    sym = Val(Symbol(str))
    return :(_strptr($sym))
end
@llvmgenerated builder function _strptr(::Val{sym})::Ptr{UInt8} where {sym}
    ptr = globalstring_ptr!(builder, String(sym))
    pointercast!(builder, ptr, convert(LLVMType, Ptr{UInt8}))
end


# it's not useful to have several threads report exceptions (interleaved output, can crash
# CUDA), so use an output lock to only have a single thread write an exception message.
# the lock goes from free (0) to claimed (1), to published (2) once the owner's index is
# visible, to finished (3) after the last message.
@inline function lock_output!(info::ExceptionInfo)
    # the lock lives in host-pinned memory, but only this GPU's threads contend for it, so
    # the default device scope is what we want (system scope is unavailable on some platforms)
    lock = info.output_lock_ptr
    state = atomic_cas!(lock, Int32(0), Int32(1))
    if state == Int32(0)
        # we just took the lock: publish our index. a fence before a relaxed store, paired
        # with a fence after a relaxed load, also orders memory before sm_70.
        info.threadIdx, info.blockIdx = threadIdx(), blockIdx()
        threadfence()
        UnsafeAtomics.store!(lock, Int32(2), UnsafeAtomics.monotonic, UnsafeAtomics.device)
        return true
    elseif state == Int32(2)
        # check whether we already have the lock
        threadfence()
        return info.threadIdx == threadIdx() && info.blockIdx == blockIdx()
    else
        # somebody else has the lock, without having published their index yet, or the
        # exception has been reported
        return false
    end
end

@device_function function report_exception(ex)
    # this is the first reporting function being called, so claim the exception
    info = kernel_state().exception_info
    if lock_output!(info)
        # override the exception type GPUCompiler deduced if the user provided a subtype
        if info.subtype != C_NULL
            ex = info.subtype
        end
        @cuprintf("ERROR: a %s was thrown during kernel execution on thread (%d, %d, %d) in block (%d, %d, %d).\n",
                  ex, threadIdx().x, threadIdx().y, threadIdx().z, blockIdx().x, blockIdx().y, blockIdx().z)
        if info.reason != C_NULL
            @cuprintf("%s\n", info.reason)
        end
        @cuprintf("Stacktrace not available, run Julia on debug level 2 for more details (by passing -g2 to the executable).\n")
    end
    return
end

@device_function function report_exception_name(ex)
    info = kernel_state().exception_info

    # this is the first reporting function being called, so claim the exception
    if lock_output!(info)
        # override the exception type GPUCompiler deduced if the user provided a subtype
        if info.subtype != C_NULL
            ex = info.subtype
        end
        @cuprintf("ERROR: a %s was thrown during kernel execution on thread (%d, %d, %d) in block (%d, %d, %d).\n",
                  ex, threadIdx().x, threadIdx().y, threadIdx().z, blockIdx().x, blockIdx().y, blockIdx().z)
        if info.reason != C_NULL
            @cuprintf("%s\n", info.reason)
        end
        @cuprintf("Stacktrace:\n")
    end
    return
end

@device_function function report_exception_frame(idx, func, file, line)
    info = kernel_state().exception_info

    if lock_output!(info)
        @cuprintf(" [%d] %s at %s:%d\n", idx, func, file, line)
    end
    return
end

@device_function function signal_exception()
    info = kernel_state().exception_info

    # finalize output
    if lock_output!(info)
        @cuprintf("\n")
        UnsafeAtomics.store!(info.output_lock_ptr, Int32(3), UnsafeAtomics.monotonic,
                             UnsafeAtomics.device)
    end

    # inform the host, which reads this after the kernel has finished
    UnsafeAtomics.store!(info.status_ptr, Int32(1), UnsafeAtomics.monotonic,
                         UnsafeAtomics.device)
    threadfence_system()

    # stop executing
    exit()

    return
end


## kernel state

struct KernelState
    exception_info::ExceptionInfo
    random_seed::UInt32
end

@inline @generated kernel_state() = GPUCompiler.kernel_state_value(KernelState)


## other

@device_function function report_oom(sz)
    @cuprintf("ERROR: Out of dynamic GPU memory (trying to allocate %d bytes)\n", sz)
    return
end
