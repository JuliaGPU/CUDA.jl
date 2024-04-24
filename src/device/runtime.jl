# GPU runtime library

import Base.Sys: WORD_SIZE

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


## exception handling

# TODO: overloads in quirks.jl still do their own printing. maybe have them set an exception
#       reason in the info structure/

struct ExceptionInfo_st
    # whether an exception has been encountered (0 -> 1)
    status::Int32

    # whether an exception is in the process of being reported (0 -> 1 -> 2)
    output_lock::Int32

    # who is reporting the exception
    thread::@NamedTuple{x::Int32,y::Int32,z::Int32}
    blockIdx::@NamedTuple{x::Int32,y::Int32,z::Int32}

    ExceptionInfo_st() = new(0, 0)
end

# to simplify use of this struct, which is passed by-reference, use property overloading
const ExceptionInfo = Ptr{ExceptionInfo_st}
@inline function Base.getproperty(info::ExceptionInfo, sym::Symbol)
    if sym === :status
        unsafe_load(convert(Ptr{Int32}, info))
    elseif sym === :output_lock
        unsafe_load(convert(Ptr{Int32}, info + 4))
    elseif sym === :output_lock_ptr
        reinterpret(LLVMPtr{Int32,AS.Generic}, info + 4)
    elseif sym === :threadIdx
        unsafe_load(convert(Ptr{@NamedTuple{x::Int32,y::Int32,z::Int32}}, info + 8))
    elseif sym === :blockIdx
        unsafe_load(convert(Ptr{@NamedTuple{x::Int32,y::Int32,z::Int32}}, info + 20))
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
    else
        setfield!(info, sym, value)
    end
end

# it's not useful to have several threads report exceptions (interleaved output, can crash
# CUDA), so use an output lock to only have a single thread write the exception message
@inline function take_output_lock(info::ExceptionInfo)
    # atomic operations on host-pinned memory are iffy, but are fine from the POV of one GPU
    if atomic_cas!(info.output_lock_ptr, Int32(0), Int32(1)) == Int32(0)
        info.threadIdx, info.blockIdx = threadIdx(), blockIdx()
        threadfence()
        return true
    else
        return false
    end
end
@inline function has_output_lock(info::ExceptionInfo)
    info.output_lock == 1 || return false

    info.threadIdx == threadIdx() || return false
    info.blockIdx == blockIdx() || return false

    return true
end

function report_exception(ex)
    # this is the first reporting function being called, so claim the exception
    if take_output_lock(kernel_state().exception_info)
        @cuprintf("""
            ERROR: a %s was thrown during kernel execution on thread (%d, %d, %d) in block (%d, %d, %d).
            For stacktrace reporting, run Julia on debug level 2 (by passing -g2 to the executable).
            """, ex, threadIdx().x, threadIdx().y, threadIdx().z, blockIdx().x, blockIdx().y, blockIdx().z)
    end
    return
end

function report_exception_name(ex)
    # this is the first reporting function being called, so claim the exception
    if take_output_lock(kernel_state().exception_info)
        @cuprintf("""
            ERROR: a %s was thrown during kernel execution on thread (%d, %d, %d) in block (%d, %d, %d).
            Stacktrace:
            """, ex, threadIdx().x, threadIdx().y, threadIdx().z, blockIdx().x, blockIdx().y, blockIdx().z)
    end
    return
end

function report_exception_frame(idx, func, file, line)
    if has_output_lock(kernel_state().exception_info)
        @cuprintf(" [%d] %s at %s:%d\n", idx, func, file, line)
    end
    return
end

function signal_exception()
    info = kernel_state().exception_info

    # finalize output
    if has_output_lock(info)
        @cuprintf("\n")
        info.output_lock = 2
    end

    # inform the host
    info.status = 1
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

function report_oom(sz)
    @cuprintf("ERROR: Out of dynamic GPU memory (trying to allocate %d bytes)\n", sz)
    return
end
