# GPU runtime library

@public precompile_runtime

"""
    precompile_runtime()

Does nothing; kept for backwards compatibility.

This used to build GPUCompiler's runtime library ahead of time. GPUCompiler now builds that
library on first use in each session, for the exact configuration of the kernel being
compiled, and caches the functions it consists of together with the rest of the compiled
code, so there is nothing left to prepare.
"""
precompile_runtime() = nothing


## exception handling

# the record of the first exception a kernel threw, in host memory that is shared by all
# kernels of a context (see `create_exceptions!`). the host checks it on synchronization.
struct ExceptionInfo_st
    # whether an exception has been encountered (0 -> 1), set once the rest is recorded
    status::Int32

    # claimed by the thread that records the exception (0 -> 1)
    claim::Int32

    # which thread threw
    threadIdx::@NamedTuple{x::Int32,y::Int32,z::Int32}
    blockIdx::@NamedTuple{x::Int32,y::Int32,z::Int32}

    # the throw site, see `GPUCompiler.uses_exception_sites`, and the table it refers to
    site::Int32
    sites::UInt32

    ExceptionInfo_st() = new(0, 0,
                             (; x=Int32(0), y=Int32(0), z=Int32(0)),
                             (; x=Int32(0), y=Int32(0), z=Int32(0)),
                             0, 0)
end

# to simplify use of this struct, which is passed by-reference, use property overloading
const ExceptionInfo = Ptr{ExceptionInfo_st}

# a field of the record, for device code: kernels see host-pinned memory in the global
# address space, and the record is 4-byte aligned. accessing it through a plain `Ptr`
# assumes neither, which makes stores go byte by byte, and atomics check at run time
# whether they target local memory.
@inline global_field(info::ExceptionInfo, ::Type{T}, offset) where {T} =
    reinterpret(LLVMPtr{T,AS.Global}, info + offset)

@inline function Base.getproperty(info::ExceptionInfo, sym::Symbol)
    if sym === :status
        unsafe_load(convert(Ptr{Int32}, info))
    elseif sym === :status_ptr
        global_field(info, Int32, 0)
    elseif sym === :claim_ptr
        global_field(info, Int32, 4)
    elseif sym === :threadIdx
        unsafe_load(convert(Ptr{@NamedTuple{x::Int32,y::Int32,z::Int32}}, info + 8))
    elseif sym === :blockIdx
        unsafe_load(convert(Ptr{@NamedTuple{x::Int32,y::Int32,z::Int32}}, info + 20))
    elseif sym === :site
        unsafe_load(convert(Ptr{Int32}, info + 32))
    elseif sym === :sites
        unsafe_load(convert(Ptr{UInt32}, info + 36))
    else
        getfield(info, sym)
    end
end
# the field values are type-asserted rather than converted: nothing reachable from
# `exception_site` may throw, not even a checked conversion, which would close an inference
# cycle through the `throw_inexacterror` overlay, pessimizing the effects of Base
# conversions like `Int32(::UInt64)`. only device code stores to the record.
@inline function Base.setproperty!(info::ExceptionInfo, sym::Symbol, value)
    Index = @NamedTuple{x::Int32,y::Int32,z::Int32}
    if sym === :status
        unsafe_store!(global_field(info, Int32, 0), value::Int32, 1, Val(4))
    elseif sym === :threadIdx
        unsafe_store!(global_field(info, Index, 8), value::Index, 1, Val(4))
    elseif sym === :blockIdx
        unsafe_store!(global_field(info, Index, 20), value::Index, 1, Val(4))
    elseif sym === :site
        unsafe_store!(global_field(info, Int32, 32), value::Int32, 1, Val(4))
    elseif sym === :sites
        unsafe_store!(global_field(info, UInt32, 36), value::UInt32, 1, Val(4))
    else
        setfield!(info, sym, value)
    end
end

# the thread that threw stops executing after this. the first one to throw records where,
# and the host looks up what was thrown there.
@device_function function exception_site(site::Int32)
    state = kernel_state()
    info = state.exception_info

    # the record lives in host-pinned memory, but only this GPU's threads contend for it,
    # so the default device scope is what we want (system scope is unavailable on some
    # platforms)
    if atomic_cas!(info.claim_ptr, Int32(0), Int32(1)) == Int32(0)
        info.site = site
        info.sites = state.exception_sites
        info.threadIdx, info.blockIdx = threadIdx(), blockIdx()

        # inform the host, which reads this after the kernel has finished. the record has
        # to be visible before the status, as the host may check while the kernel runs.
        threadfence_system()
        UnsafeAtomics.store!(info.status_ptr, Int32(1), UnsafeAtomics.monotonic,
                             UnsafeAtomics.device)
        threadfence_system()
    end

    # stop executing
    exit()

    return
end


## kernel state

struct KernelState
    exception_info::ExceptionInfo
    # the table of throw sites that the kernel reports exceptions with, see
    # `create_exceptions!`
    exception_sites::UInt32
    random_seed::UInt32
end

@inline @generated kernel_state() = GPUCompiler.kernel_state_value(KernelState)


## other

# GPUCompiler's runtime calls this when device `malloc` fails, right before it throws. throw
# here instead, so that the exception says which memory ran out.
@device_function function report_oom(sz)
    GPUCompiler.@gputhrow "OutOfMemoryError" "Device heap allocation failed. To increase the heap, call `CUDA.limit!(CUDA.LIMIT_MALLOC_HEAP_SIZE, bytes)` before a kernel that uses device `malloc` or `free` is loaded in this context."
end
