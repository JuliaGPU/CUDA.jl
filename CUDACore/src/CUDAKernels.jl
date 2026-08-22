module CUDAKernels

using ..CUDACore
using ..CUDACore: @device_override, default_memory, UnifiedMemory, GPUArrays

import KernelInterface as KI

import StaticArrays

import Adapt

## back-end

# export CUDABackend

"""
    CUDABackend(; prefer_blocks=false, always_inline=false, fastmath=false)

KernelAbstractions backend for CUDA. `fastmath=true` enables the same floating-point
optimizations as `@cuda fastmath=true`, including flushing `Float32` subnormals to zero.
The default follows Julia's `--math-mode` setting.
"""
struct CUDABackend <: KI.GPU
    prefer_blocks::Bool
    always_inline::Bool
    fastmath::Bool
end

CUDABackend(; prefer_blocks=false, always_inline=false,
              fastmath=Base.JLOptions().fast_math == 1) =
    CUDABackend(prefer_blocks, always_inline, fastmath)
CUDABackend(prefer_blocks, always_inline) = CUDABackend(; prefer_blocks, always_inline)

@inline KI.allocate(::CUDABackend, ::Type{T}, dims::Tuple; unified::Bool = false) where T = CuArray{T, length(dims), unified ? UnifiedMemory : default_memory}(undef, dims)
@inline KI.zeros(::CUDABackend, ::Type{T}, dims::Tuple; unified::Bool = false) where T = fill!(CuArray{T, length(dims), unified ? UnifiedMemory : default_memory}(undef, dims), zero(T))
@inline KI.ones(::CUDABackend, ::Type{T}, dims::Tuple; unified::Bool = false) where T = fill!(CuArray{T, length(dims), unified ? UnifiedMemory : default_memory}(undef, dims), one(T))

KI.get_backend(::CuArray) = CUDABackend()
KI.synchronize(::CUDABackend) = synchronize()

KI.functional(::CUDABackend) = CUDACore.functional()

KI.supports_unified(::CUDABackend) = true

Adapt.adapt_storage(::CUDABackend, a::AbstractArray) = Adapt.adapt(CuArray, a)
Adapt.adapt_storage(::CUDABackend, a::Union{CuArray,GPUArrays.AbstractGPUSparseArray}) = a

## memory operations

function KI.copyto!(::CUDABackend, A, B)
    GC.@preserve A B begin
        destptr = pointer(A)
        srcptr  = pointer(B)
        N       = length(A)
        unsafe_copyto!(destptr, srcptr, N, async=true)
    end
    return A
end

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
end

KI.argconvert(::CUDABackend, arg) = cudaconvert(arg)

function KI.kernel_function(backend::CUDABackend, f::F, tt::TT=Tuple{}; name=nothing, kwargs...) where {F,TT}
    kern = cufunction(f, tt; name, fastmath=backend.fastmath, kwargs...)
    KI.Kernel{CUDABackend, typeof(kern)}(CUDABackend(), kern)
end

function (obj::KI.Kernel{CUDABackend})(args...; numworkgroups=(), workgroupsize=(), ndrange=(), max_work_group_size=typemax(Int))
    KI.check_launch_args(numworkgroups, workgroupsize, ndrange)
    prod(ndrange) == 0 && return nothing

    blocks, threads = KI.auto_launch_sizes(obj, numworkgroups, workgroupsize, ndrange, max_work_group_size)

    obj.kern(args...; threads, blocks)
    return nothing
end


function KI.kernel_max_work_group_size(kernel::KI.Kernel{<:CUDABackend}; max_work_items::Int=typemax(Int))::Int
    kernel_config = launch_configuration(kernel.kern.fun)

    Int(min(kernel_config.threads, max_work_items))
end
function KI.max_work_group_size(::CUDABackend)::Int
    Int(attribute(device(), CUDACore.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))
end
function KI.sub_group_size(::CUDABackend)::Int
    warpsize(device())
end
function KI.multiprocessor_count(::CUDABackend)::Int
    Int(attribute(device(), CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
end

KI.shfl_down_types(::CUDABackend) = DataType[Bool,
                                             UInt8, UInt16, UInt32, UInt64, UInt128,
                                             Int8, Int16, Int32, Int64, Int128,
                                             Float16, Float32, Float64,
                                             ComplexF16, ComplexF32, ComplexF64]

## indexing

@device_override @inline function KI.get_local_id(::Type{T}) where {T}
    return (; x = T(threadIdx().x), y = T(threadIdx().y), z = T(threadIdx().z))
end

@device_override @inline function KI.get_group_id(::Type{T}) where {T}
    return (; x = T(blockIdx().x), y = T(blockIdx().y), z = T(blockIdx().z))
end

@device_override @inline function KI.get_global_id(::Type{T}) where {T}
    return (; x = T((blockIdx().x-1)*blockDim().x + threadIdx().x), y = T((blockIdx().y-1)*blockDim().y + threadIdx().y), z = T((blockIdx().z-1)*blockDim().z + threadIdx().z))
end

@device_override @inline function KI.get_local_size(::Type{T}) where {T}
    return (; x = T(blockDim().x), y = T(blockDim().y), z = T(blockDim().z))
end

@device_override @inline function KI.get_num_groups(::Type{T}) where {T}
    return (; x = T(gridDim().x), y = T(gridDim().y), z = T(gridDim().z))
end

@device_override @inline function KI.get_global_size(::Type{T}) where {T}
    return (; x = T(blockDim().x * gridDim().x), y = T(blockDim().y * gridDim().y), z = T(blockDim().z * gridDim().z))
end

@device_override KI.get_sub_group_size() = UInt32(warpsize())

@device_override KI.get_max_sub_group_size() = UInt32(warpsize())

@device_override KI.get_num_sub_groups() = UInt32(prod(blockDim()) ÷ warpsize())

@device_override KI.get_sub_group_id() = UInt32(((threadIdx().x - 1) + blockDim().x * (threadIdx().y - 1) + blockDim().x * blockDim().y * (threadIdx().z - 1)) ÷ warpsize()) + 0x1

@device_override KI.get_sub_group_local_id() = UInt32(laneid())


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
