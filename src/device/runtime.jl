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
@generated function _strptr(::Val{sym}) where {sym}
    str = String(sym)
    @dispose ctx=Context() begin
        T_pint8 = LLVM.PointerType(LLVM.Int8Type())
        T_ptr = convert(LLVMType, Ptr{UInt8})

        # create function
        llvm_f, llvm_ft = create_function(T_ptr)

        # generate IR
        @dispose builder=IRBuilder() begin
            entry = BasicBlock(llvm_f, "entry")
            position!(builder, entry)

            ptr = globalstring_ptr!(builder, str)
            jlptr = ptrtoint!(builder, ptr, T_ptr)
            ret!(builder, jlptr)
        end

        call_function(llvm_f, Ptr{UInt8}, Tuple{})
    end
end


# it's not useful to have several threads report exceptions (interleaved output, can crash
# CUDA), so use an output lock to only have a single thread write an exception message
@inline function lock_output!(info::ExceptionInfo)
    # atomic operations on host-pinned memory are iffy, but are fine from the POV of one GPU
    if atomic_cas!(info.output_lock_ptr, Int32(0), Int32(1)) == Int32(0)
        # we just took the lock, note our index
        info.threadIdx, info.blockIdx = threadIdx(), blockIdx()
        threadfence()
        return true
    elseif info.output_lock == 1 && info.threadIdx == threadIdx() && info.blockIdx == blockIdx()
        # we already have the lock
        return true
    else
        # somebody else has the lock
        return false
    end
end

function report_exception(ex)
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

function report_exception_name(ex)
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

function report_exception_frame(idx, func, file, line)
    info = kernel_state().exception_info

    if lock_output!(info)
        @cuprintf(" [%d] %s at %s:%d\n", idx, func, file, line)
    end
    return
end

function signal_exception()
    info = kernel_state().exception_info

    # finalize output
    if lock_output!(info)
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
