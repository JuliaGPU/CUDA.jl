module CUDAKernels

using ..CUDACore
using ..CUDACore: @device_override, default_memory, UnifiedMemory, GPUArrays, i32, compute_capability
using GPUToolbox: @sv_str

import KernelInterface as KI

import Adapt

## back-end

"""
    CUDABackend(; prefer_blocks=false, always_inline=false, fastmath=false)

KernelAbstractions backend for CUDA. `fastmath=true` enables the same floating-point
optimizations as `@cuda fastmath=true`, including flushing `Float32` subnormals to zero.
The default follows Julia's `--math-mode` setting.
"""
struct CUDABackend <: KI.Backend
    prefer_blocks::Bool
    always_inline::Bool
    fastmath::Bool
end

CUDABackend(; prefer_blocks=false, always_inline=false,
              fastmath=Base.JLOptions().fast_math == 1) =
    CUDABackend(prefer_blocks, always_inline, fastmath)
CUDABackend(prefer_blocks, always_inline) = CUDABackend(; prefer_blocks, always_inline)

@inline KI.allocate(::CUDABackend, ::Type{T}, dims::Tuple; unified::Bool = false) where T = CuArray{T, length(dims), unified ? UnifiedMemory : default_memory}(undef, dims)

KI.get_backend(::CuArray) = CUDABackend()
KI.synchronize(::CUDABackend) = synchronize()

KI.functional(::CUDABackend) = CUDACore.functional()

KI.supports_unified(::CUDABackend) = true
KI.supports_float64(::CUDABackend) = true
KI.supports_atomics(::CUDABackend) = true
KI.supports_subgroups(::CUDABackend) = true
# warps are formed from consecutive linear thread indices, and are independent
KI.supports_linear_subgroups(::CUDABackend) = true
KI.supports_independent_subgroups(::CUDABackend) = true
# the primitive types CUDA's warp shuffles support (decomposing them into 32-bit shuffles);
# KernelInterface shuffles other `isbits` types, e.g. `Complex`, field by field
const ShuffleTypes = Union{Bool, Base.BitInteger, Base.IEEEFloat}
KI.supports_shuffle(::CUDABackend, ::Type{<:ShuffleTypes}) = true

Adapt.adapt_storage(::CUDABackend, a::AbstractArray) = Adapt.adapt(CuArray, a)
Adapt.adapt_storage(::CUDABackend, a::Union{CuArray,GPUArrays.AbstractGPUSparseArray}) = a

## memory operations

# dense arrays, and contiguous views of them
const ContiguousArray{T} = Union{DenseArray{T}, Base.FastContiguousSubArray{T, <:Any, <:DenseArray}}
on_device(A::ContiguousArray) = parent(A) isa CuArray

function KI.copyto!(::CUDABackend, A::ContiguousArray{T}, B::ContiguousArray{T}) where {T}
    length(A) == length(B) ||
        throw(ArgumentError("Arrays must have the same length, got $(length(A)) and $(length(B))"))
    if isbitstype(T) && (on_device(A) || on_device(B))
        GC.@preserve A B CUDACore.with_managed_arrays(A, B) do
            unsafe_copyto!(pointer(A), pointer(B), length(A), async=true)
        end
    else
        # host-to-host copies, and bits unions, whose type tags are stored separately
        copyto!(A, B)
    end
    return A
end
KI.copyto!(::CUDABackend, A, B) =
    throw(ArgumentError("KernelInterface.copyto! only supports contiguous arrays of the same element type, got $(typeof(A)) and $(typeof(B))"))

KI.unsafe_free!(A::CuArray) = CUDACore.unsafe_free!(A)

function KI.pagelock!(::CUDABackend, A::Array)
    CUDACore.pin(A)
    return nothing
end

## device operations

function KI.ndevices(::CUDABackend)
    return Int(ndevices())
end

function KI.device(::CUDABackend)::Int
    deviceid(CUDACore.active_state().device) + 1
end

function KI.device!(backend::CUDABackend, id::Int)
    if !(0 < id <= KI.ndevices(backend))
        throw(ArgumentError("Device id $id out of bounds."))
    end
    device!(id - 1)
    return
end

KI.device(::CUDABackend, A::CuArray) = deviceid(CUDACore.device(A)) + 1

KI.argconvert(::CUDABackend, arg) = cudaconvert(arg)

# a compiled kernel, and the callable it was compiled from. that one is converted again at
# every launch, like the arguments, which keeps the arrays it captures alive and registers
# their managed memory.
struct CompiledKernel{K,F}
    kernel::K
    f::F
end

function KI.kernel_function(backend::CUDABackend, f::F, tt::TT=Tuple{}; name=nothing, kwargs...) where {F,TT}
    kernel = cufunction(cudaconvert(f), tt; name, backend.always_inline, backend.fastmath, kwargs...)
    KI.Kernel(backend, CompiledKernel(kernel, f))
end

# passes the arguments on as a tuple, like calling the `HostKernel` does
function KI.launch(obj::KI.Kernel{CUDABackend}, groups::Dims{3}, items::Dims{3},
                   args::Tuple; kwargs...)
    # KernelInterface has validated the launch geometry
    if haskey(kwargs, :threads) || haskey(kwargs, :blocks)
        throw(ArgumentError("KernelInterface kernels take `numgroups`, `workgroupsize` or `ndrange`, not `threads` or `blocks`"))
    end
    call = CUDACore.kernel_call(CUDACore.LLVMBackend(), obj.kern.f, args)
    CUDACore.kernel_launch(obj.kern.kernel, call; threads=items, blocks=groups, kwargs...)
    return
end

# these queries are compiled for every kernel, so they forward to functions of the
# `CuFunction` that are compiled only once
KI.max_work_group_size(kernel::KI.Kernel{CUDABackend})::Int =
    max_threads_per_block(kernel.kern.kernel.fun)
KI.launch_configuration(kernel::KI.Kernel{CUDABackend}; nitems::Union{Integer,Nothing}=nothing,
                        max_work_group_size::Integer=typemax(Int)) =
    (; workgroupsize=launch_threads(kernel.kern.kernel.fun, kernel.backend.prefer_blocks,
                                    nitems, max_work_group_size))

max_threads_per_block(fun::CuFunction) =
    Int(attributes(fun)[CUDACore.FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK])
function launch_threads(fun::CuFunction, prefer_blocks::Bool, nitems, max_work_group_size)
    max_threads = min(max_work_group_size, something(nitems, typemax(Int)), typemax(Int32))
    config = launch_configuration(fun; max_threads)
    threads = Int(config.threads)
    if prefer_blocks && nitems !== nothing
        # prefer blocks over threads: at least as many blocks as the occupancy API suggests
        # XXX: some kernels perform much better with all blocks active
        blocks = max(cld(nitems, threads), Int(config.blocks))
        threads = cld(nitems, blocks)
    end
    return threads
end
# these limits are the same for every supported device, so don't query them on every launch
KI.max_work_group_size(::CUDABackend)::Int = 1024
KI.max_work_group_dims(::CUDABackend)::NTuple{3, Int} = (1024, 1024, 64)
KI.max_num_groups(::CUDABackend)::NTuple{3, Int} = (Int(typemax(Int32)), 65535, 65535)
function KI.sub_group_size(::CUDABackend)::Int
    warpsize(device())
end
function KI.multiprocessor_count(::CUDABackend)::Int
    Int(attribute(device(), CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
end

## indexing

# computed with `% T`, which unlike `T(x)` has no error path

@device_override @inline function KI.get_local_id(::Type{T}) where {T}
    return (; x = threadIdx().x % T, y = threadIdx().y % T, z = threadIdx().z % T)
end

@device_override @inline function KI.get_group_id(::Type{T}) where {T}
    return (; x = blockIdx().x % T, y = blockIdx().y % T, z = blockIdx().z % T)
end

@device_override @inline function KI.get_local_size(::Type{T}) where {T}
    return (; x = blockDim().x % T, y = blockDim().y % T, z = blockDim().z % T)
end

@device_override @inline function KI.get_num_groups(::Type{T}) where {T}
    return (; x = gridDim().x % T, y = gridDim().y % T, z = gridDim().z % T)
end

# the warp size is 32 on every NVIDIA GPU (CUDA.jl's warp intrinsics assume so too), so it is
# a constant rather than a read of `%WARP_SZ`, which the compiler can't fold
const WARP_SIZE = 32i32

# warps are formed from consecutive linear thread indices, x fastest. the queries are computed
# with unsigned 32-bit integers (a block has at most 1024 threads), so that the divisions by
# the warp size are shifts, and are inlined, as they are cheap.
@inline function linear_thread_id()
    x = (threadIdx().x - 1i32) % UInt32
    y = (threadIdx().y - 1i32) % UInt32
    z = (threadIdx().z - 1i32) % UInt32
    return (z * (blockDim().y % UInt32) + y) * (blockDim().x % UInt32) + x
end

@inline block_threads() = (blockDim().x * blockDim().y * blockDim().z) % UInt32

# the last warp of a block can be partial
@device_override @inline KI.get_sub_group_size(::Type{T}) where {T} =
    min(0x00000020, block_threads() - (linear_thread_id() & ~0x0000001f)) % T

@device_override @inline KI.get_max_sub_group_size(::Type{T}) where {T} = WARP_SIZE % T

@device_override @inline KI.get_num_sub_groups(::Type{T}) where {T} =
    ((block_threads() + 0x1f) >> 0x5) % T

@device_override @inline KI.get_sub_group_id(::Type{T}) where {T} =
    ((linear_thread_id() >> 0x5) + 0x1) % T

@device_override @inline KI.get_sub_group_local_id(::Type{T}) where {T} = laneid() % T

## shared and scratch memory

@device_override @inline function KI.localmemory(::Type{T}, ::Val{Dims}) where {T, Dims}
    CuStaticSharedArray(T, Dims)
end

## synchronization and printing

@device_override @inline function KI.barrier()
    sync_threads()
end

@device_override @inline function KI.sub_group_barrier()
    sync_warp()
end

# the lanes and offsets are truncated to the `UInt32` the intrinsics take, so that values out
# of range don't throw an `InexactError` but give an unspecified value. `shfl_sync` takes a
# 1-based lane and subtracts one from it, so wrap the lane to 1:32 (as PTX would wrap the
# 0-based one to 0:31).
@device_override @inline KI.shfl(val::T, lane::Integer) where {T <: ShuffleTypes} =
    shfl_sync(FULL_MASK, val, shfl_lane(lane))

@inline shfl_lane(lane) = ((lane - one(lane)) % UInt32 & 0x1f) + 0x1

# `shfl.sync` returns the thread's own value where the source lane is past the warp (or the
# segment of `width` lanes), as KernelInterface requires, but only uses the low 5 bits of the
# offset, so offsets of 32 and more (for which every thread gets its own value) become 0. they
# are compared before they are narrowed, so that also wide offsets do.
@inline shfl_offset(offset) = ifelse((offset >= 0) & (offset < 32), offset % UInt32, 0x00000000)

@device_override @inline KI.shfl_down(val::T, offset::Integer) where {T <: ShuffleTypes} =
    shfl_down_sync(FULL_MASK, val, shfl_offset(offset))

@device_override @inline KI.shfl_up(val::T, offset::Integer) where {T <: ShuffleTypes} =
    shfl_up_sync(FULL_MASK, val, shfl_offset(offset))

# the mask is below the warp size
@device_override @inline KI.shfl_xor(val::T, mask::Integer) where {T <: ShuffleTypes} =
    shfl_xor_sync(FULL_MASK, val, mask % UInt32)

# the shuffles within segments of `width` lanes, a single `shfl.sync` (rather than KI's
# fallbacks, a `shfl.sync` from a lane computed with a few integer operations). `shfl.sync`
# takes the lane modulo `width`, and reads from the thread itself where the source lane is
# past the end of the segment, or before its start for `shfl_up`. KernelInterface requires
# the mask of `shfl_xor` to be below `width`, so it stays within the segment.
@device_override @inline KI.shfl(val::T, lane::Integer, width::Integer) where {T <: ShuffleTypes} =
    shfl_sync(FULL_MASK, val, shfl_lane(lane), width % UInt32)

@device_override @inline KI.shfl_down(val::T, offset::Integer, width::Integer) where {T <: ShuffleTypes} =
    shfl_down_sync(FULL_MASK, val, shfl_offset(offset), width % UInt32)

@device_override @inline KI.shfl_up(val::T, offset::Integer, width::Integer) where {T <: ShuffleTypes} =
    shfl_up_sync(FULL_MASK, val, shfl_offset(offset), width % UInt32)

@device_override @inline KI.shfl_xor(val::T, mask::Integer, width::Integer) where {T <: ShuffleTypes} =
    shfl_xor_sync(FULL_MASK, val, mask % UInt32, width % UInt32)

@device_override KI.sub_group_any(pred::Bool) = vote_any_sync(FULL_MASK, pred)

@device_override KI.sub_group_all(pred::Bool) = vote_all_sync(FULL_MASK, pred)

# bit `i - 1` for the thread with (1-based) lane `i`
@device_override KI.sub_group_ballot(pred::Bool) = UInt64(vote_ballot_sync(FULL_MASK, pred))

# `redux.sync` reduces 32-bit integers from sm_80 on, over the threads of the warp (as for the
# shuffles, lanes past a partial last warp don't exist and aren't waited for). not over
# `activemask()`: all threads of the warp execute `sub_group_reduce` together, but needn't be
# converged when they read the active mask.
# the operators are commutative, so the order of the lanes doesn't matter. other operators,
# types and devices use KernelInterface's fallback.
for (op, T, name) in ((:+, Int32, "add"), (:+, UInt32, "add"),
                      (:min, Int32, "min"), (:max, Int32, "max"),
                      (:min, UInt32, "umin"), (:max, UInt32, "umax"),
                      (:&, Int32, "and"), (:&, UInt32, "and"),
                      (:|, Int32, "or"), (:|, UInt32, "or"),
                      (:⊻, Int32, "xor"), (:⊻, UInt32, "xor"))
    intr = "llvm.nvvm.redux.sync.$name"
    @eval @device_override @inline function KI.sub_group_reduce(op::typeof($op), val::$T)
        if compute_capability() >= sv"8.0"
            return ccall($intr, llvmcall, $T, ($T, UInt32), val, FULL_MASK)
        else
            return invoke(KI.sub_group_reduce, Tuple{Any, Any}, op, val)
        end
    end
end

@device_override @inline function KI._print(args...)
    CUDACore._cuprint(args...)
end

## events

function KI.record_event(::CUDABackend)
    ev = CuEvent(CUDACore.EVENT_DISABLE_TIMING)
    record(ev, stream())
    return ev
end

function KI.wait_event(::CUDABackend, ev::CuEvent)
    CUDACore.wait(ev, stream())
    return
end

## other

function KI.priority!(::CUDABackend, prio::Symbol)
    if !(prio in (:high, :normal, :low))
        error("priority must be one of :high, :normal, :low")
    end
    CUDACore.priority!(prio)
    return nothing
end

end
