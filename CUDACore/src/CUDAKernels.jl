module CUDAInterface

using ..CUDACore
using ..CUDACore: @device_override, default_memory, UnifiedMemory, GPUArrays

import KernelInterface as KI

import StaticArrays

import Adapt

## back-end

export CUDABackend

struct CUDABackend <: KI.GPU
    prefer_blocks::Bool
    always_inline::Bool
end

CUDABackend(; prefer_blocks=false, always_inline=false) = CUDABackend(prefer_blocks, always_inline)

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

# function (obj::KA.Kernel{CUDABackend})(args...; ndrange=nothing, workgroupsize=nothing)
#     backend = KA.backend(obj)

#     ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, workgroupsize)
#     # this might not be the final context, since we may tune the workgroupsize
#     ctx = KA.mkcontext(obj, ndrange, iterspace)

#     # If the kernel is statically sized we can tell the compiler about that
#     if KA.workgroupsize(obj) <: KA.StaticSize
#         maxthreads = prod(KA.get(KA.workgroupsize(obj)))
#     else
#         maxthreads = nothing
#     end

#     call = CUDACore.kernel_call(obj.f, (ctx, args...))
#     kernel = CUDACore.kernel_compile(call; always_inline=backend.always_inline, maxthreads)

#     # figure out the optimal workgroupsize automatically
#     if KA.workgroupsize(obj) <: KA.DynamicSize && workgroupsize === nothing
#         config = CUDACore.launch_configuration(kernel.fun; max_threads=prod(ndrange))
#         if backend.prefer_blocks
#             # Prefer blocks over threads
#             threads = min(prod(ndrange), config.threads)
#             # XXX: Some kernels performs much better with all blocks active
#             cu_blocks = max(cld(prod(ndrange), threads), config.blocks)
#             threads = cld(prod(ndrange), cu_blocks)
#         else
#             threads = config.threads
#         end

#         workgroupsize = threads_to_workgroupsize(threads, ndrange)
#         iterspace, dynamic = KA.partition(obj, ndrange, workgroupsize)
#         ctx = KA.mkcontext(obj, ndrange, iterspace)
#         call = CUDACore.rebind(call, ctx, 1)
#     end

#     blocks = length(KA.blocks(iterspace))
#     threads = length(KA.workitems(iterspace))

#     if blocks == 0
#         return nothing
#     end

#     # Launch kernel
#     CUDACore.kernel_launch(kernel, call; threads, blocks)

#     return nothing
# end

KI.argconvert(::CUDABackend, arg) = cudaconvert(arg)

function KI.kernel_function(::CUDABackend, f::F, tt::TT=Tuple{}; name=nothing, kwargs...) where {F,TT}
    kern = cufunction(f, tt; name, kwargs...)
    KI.Kernel{CUDABackend, typeof(kern)}(CUDABackend(), kern)
end

function (obj::KI.Kernel{CUDABackend})(args...; numworkgroups=(), workgroupsize=(), ndrange=(), max_work_group_size=typemax(Int))
    KI.check_launch_args(numworkgroups, workgroupsize, ndrange)
    prod(ndrange) == 0 && return nothing

    numworkgroups, workgroupsize = KI.auto_launch_sizes(obj, numworkgroups, workgroupsize, ndrange, max_work_group_size)

    obj.kern(args...; threads=workgroupsize, blocks=numworkgroups)
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

## COV_EXCL_START
@device_override @inline function KI.get_local_id()
    return (; x = Int(threadIdx().x), y = Int(threadIdx().y), z = Int(threadIdx().z))
end

@device_override @inline function KI.get_group_id()
    return (; x = Int(blockIdx().x), y = Int(blockIdx().y), z = Int(blockIdx().z))
end

@device_override @inline function KI.get_global_id()
    return (; x = Int((blockIdx().x-1)*blockDim().x + threadIdx().x), y = Int((blockIdx().y-1)*blockDim().y + threadIdx().y), z = Int((blockIdx().z-1)*blockDim().z + threadIdx().z))
end

@device_override @inline function KI.get_local_size()
    return (; x = Int(blockDim().x), y = Int(blockDim().y), z = Int(blockDim().z))
end

@device_override @inline function KI.get_num_groups()
    return (; x = Int(gridDim().x), y = Int(gridDim().y), z = Int(gridDim().z))
end

@device_override @inline function KI.get_global_size()
    return (; x = Int(blockDim().x * gridDim().x), y = Int(blockDim().y * gridDim().y), z = Int(blockDim().z * gridDim().z))
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

## COV_EXCL_STOP

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

    @debug "Switching default stream" flags priority
    new_stream = CuStream(; flags, priority)
    CUDACore.wait(event, new_stream)
    stream!(new_stream)
    return nothing
end

end
