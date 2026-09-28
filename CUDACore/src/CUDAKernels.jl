module CUDAKernels

using ..CUDACore
using ..CUDACore: @device_override, default_memory, UnifiedMemory, GPUArrays, i32

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
# `shfl_down_sync` decomposes other types into 32-bit shuffles
KI.supports_shuffle(::CUDABackend, ::Type{T}) where {T} =
    T <: Union{Bool, Base.BitInteger, Base.IEEEFloat, Complex{<:Union{Base.BitInteger, Base.IEEEFloat}}}

Adapt.adapt_storage(::CUDABackend, a::AbstractArray) = Adapt.adapt(CuArray, a)
Adapt.adapt_storage(::CUDABackend, a::Union{CuArray,GPUArrays.AbstractGPUSparseArray}) = a

## memory operations

function KI.copyto!(::CUDABackend, A::DenseArray{T}, B::DenseArray{T}) where {T}
    length(A) == length(B) ||
        throw(ArgumentError("Arrays must have the same length, got $(length(A)) and $(length(B))"))
    if isbitstype(T) && (A isa CuArray || B isa CuArray)
        GC.@preserve A B begin
            unsafe_copyto!(pointer(A), pointer(B), length(A), async=true)
        end
    else
        # host-to-host copies, and bits unions, whose type tags are stored separately
        copyto!(A, B)
    end
    return A
end
KI.copyto!(::CUDABackend, A, B) =
    throw(ArgumentError("KernelInterface.copyto! only supports dense arrays of the same element type, got $(typeof(A)) and $(typeof(B))"))

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

function KI.kernel_function(backend::CUDABackend, f::F, tt::TT=Tuple{}; name=nothing, kwargs...) where {F,TT}
    kern = cufunction(f, tt; name, backend.always_inline, backend.fastmath, kwargs...)
    KI.Kernel(backend, kern)
end

# specialize on the arguments, which are only passed through (Julia doesn't otherwise)
function KI.launch(obj::KI.Kernel{CUDABackend}, groups::Dims{3}, items::Dims{3},
                   args::Vararg{Any,N}; kwargs...) where {N}
    obj.kern(args...; threads=items, blocks=groups, kwargs...)
    return
end

KI.max_work_group_size(kernel::KI.Kernel{CUDABackend})::Int = CUDACore.maxthreads(kernel.kern)
function KI.launch_configuration(kernel::KI.Kernel{CUDABackend}; max_work_group_size::Integer=typemax(Int))
    config = launch_configuration(kernel.kern.fun; max_threads=min(max_work_group_size, typemax(Int32)))
    return (; workgroupsize=Int(config.threads))
end
function KI.max_work_group_size(::CUDABackend)::Int
    Int(attribute(device(), CUDACore.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))
end
# these limits are the same for every supported device, so don't query them on every launch
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

@device_override KI.get_sub_group_size(::Type{T}) where {T} = active_sub_group_size() % T

@device_override KI.get_max_sub_group_size(::Type{T}) where {T} = warpsize() % T

@device_override KI.get_num_sub_groups(::Type{T}) where {T} = cld(prod(blockDim()), warpsize()) % T

@device_override KI.get_sub_group_id(::Type{T}) where {T} = (linear_thread_id() ÷ warpsize() + 1i32) % T

@device_override KI.get_sub_group_local_id(::Type{T}) where {T} = laneid() % T

# warps are formed from consecutive linear thread indices
@inline function linear_thread_id()
    return (threadIdx().x - 1i32) +
           (threadIdx().y - 1i32) * blockDim().x +
           (threadIdx().z - 1i32) * blockDim().x * blockDim().y
end

# the last warp of a block can be partial
@inline function active_sub_group_size()
    threads = blockDim().x * blockDim().y * blockDim().z
    return min(warpsize(), threads - (linear_thread_id() ÷ warpsize()) * warpsize())
end

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

@device_override function KI.shfl_down(val::T, offset::Integer) where T
    shfl_down_sync(0xffffffff, val, offset)
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

    range = priority_range()
    # 0:-1:-5
    # lower number is higher priority, default is 0
    # there is no "low"
    if prio === :high
        priority = last(range)
    elseif prio === :normal || prio === :low
        priority = first(range)
    end

    old_stream = stream()
    r_flags = Ref{Cuint}()
    CUDACore.cuStreamGetFlags(old_stream, r_flags)
    flags = CUDACore.CUstream_flags_enum(r_flags[])

    event = CuEvent(CUDACore.EVENT_DISABLE_TIMING)
    record(event, old_stream)

    @debug "Switching default stream" flags priority _group=:CUDA
    new_stream = CuStream(; flags, priority)
    CUDACore.wait(event, new_stream)
    stream!(new_stream)
    return nothing
end

end
