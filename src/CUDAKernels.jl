module CUDAKernels

using ..CUDA
using ..CUDA: @device_override, CUSPARSE, default_memory, UnifiedMemory

import KernelAbstractions as KA
import KernelAbstractions: KernelIntrinsics as KI

import StaticArrays
import SparseArrays: AbstractSparseArray

import Adapt

## back-end

export CUDABackend

struct CUDABackend <: KA.GPU
    prefer_blocks::Bool
    always_inline::Bool
end

CUDABackend(; prefer_blocks=false, always_inline=false) = CUDABackend(prefer_blocks, always_inline)

@inline KA.allocate(::CUDABackend, ::Type{T}, dims::Tuple; unified::Bool = false) where T = CuArray{T, length(dims), unified ? UnifiedMemory : default_memory}(undef, dims)
@inline KA.zeros(::CUDABackend, ::Type{T}, dims::Tuple; unified::Bool = false) where T = fill!(CuArray{T, length(dims), unified ? UnifiedMemory : default_memory}(undef, dims), zero(T))
@inline KA.ones(::CUDABackend, ::Type{T}, dims::Tuple; unified::Bool = false) where T = fill!(CuArray{T, length(dims), unified ? UnifiedMemory : default_memory}(undef, dims), one(T))

KA.get_backend(::CuArray) = CUDABackend()
KA.get_backend(::CUSPARSE.AbstractCuSparseArray) = CUDABackend()
KA.synchronize(::CUDABackend) = synchronize()

KA.functional(::CUDABackend) = CUDA.functional()

KA.supports_unified(::CUDABackend) = true

Adapt.adapt_storage(::CUDABackend, a::AbstractArray) = Adapt.adapt(CuArray, a)
Adapt.adapt_storage(::CUDABackend, a::Union{CuArray,CUSPARSE.AbstractCuSparseArray}) = a
Adapt.adapt_storage(::KA.CPU, a::Union{CuArray,CUSPARSE.AbstractCuSparseArray}) = Adapt.adapt(Array, a)

## memory operations

function KA.copyto!(::CUDABackend, A, B)
    GC.@preserve A B begin
        destptr = pointer(A)
        srcptr  = pointer(B)
        N       = length(A)
        unsafe_copyto!(destptr, srcptr, N, async=true)
    end
    return A
end

function KA.pagelock!(::CUDABackend, A::Array)
    CUDA.pin(A)
    return nothing
end

## device operations

function KA.ndevices(::CUDABackend)
    return Int(ndevices())
end

function KA.device(::CUDABackend)::Int
    deviceid(CUDA.active_state().device) + 1
end

function KA.device!(backend::CUDABackend, id::Int)
    if !(0 < id <= KA.ndevices(backend))
        throw(ArgumentError("Device id $id out of bounds."))
    end
    device!(id - 1)
end

## kernel launch

function KA.mkcontext(kernel::KA.Kernel{CUDABackend}, _ndrange, iterspace)
    KA.CompilerMetadata{KA.ndrange(kernel), KA.DynamicCheck}(_ndrange, iterspace)
end

function KA.launch_config(kernel::KA.Kernel{CUDABackend}, ndrange, workgroupsize)
    if ndrange isa Integer
        ndrange = (ndrange,)
    end
    if workgroupsize isa Integer
        workgroupsize = (workgroupsize, )
    end

    # partition checked that the ndrange's agreed
    if KA.ndrange(kernel) <: KA.StaticSize
        ndrange = nothing
    end

    iterspace, dynamic = if KA.workgroupsize(kernel) <: KA.DynamicSize &&
        workgroupsize === nothing
        # use ndrange as preliminary workgroupsize for autotuning
        KA.partition(kernel, ndrange, ndrange)
    else
        KA.partition(kernel, ndrange, workgroupsize)
    end

    return ndrange, workgroupsize, iterspace, dynamic
end

function threads_to_workgroupsize(threads, ndrange)
    total = 1
    return map(ndrange) do n
        x = min(div(threads, total), n)
        total *= x
        return x
    end
end

function (obj::KA.Kernel{CUDABackend})(args...; ndrange=nothing, workgroupsize=nothing)
    backend = KA.backend(obj)

    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, workgroupsize)
    # this might not be the final context, since we may tune the workgroupsize
    ctx = KA.mkcontext(obj, ndrange, iterspace)

    # If the kernel is statically sized we can tell the compiler about that
    if KA.workgroupsize(obj) <: KA.StaticSize
        maxthreads = prod(KA.get(KA.workgroupsize(obj)))
    else
        maxthreads = nothing
    end

    kernel = @cuda launch=false always_inline=backend.always_inline maxthreads=maxthreads obj.f(ctx, args...)

    # figure out the optimal workgroupsize automatically
    if KA.workgroupsize(obj) <: KA.DynamicSize && workgroupsize === nothing
        config = CUDA.launch_configuration(kernel.fun; max_threads=prod(ndrange))
        if backend.prefer_blocks
            # Prefer blocks over threads
            threads = min(prod(ndrange), config.threads)
            # XXX: Some kernels performs much better with all blocks active
            cu_blocks = max(cld(prod(ndrange), threads), config.blocks)
            threads = cld(prod(ndrange), cu_blocks)
        else
            threads = config.threads
        end

        workgroupsize = threads_to_workgroupsize(threads, ndrange)
        iterspace, dynamic = KA.partition(obj, ndrange, workgroupsize)
        ctx = KA.mkcontext(obj, ndrange, iterspace)
    end

    blocks = length(KA.blocks(iterspace))
    threads = length(KA.workitems(iterspace))

    if blocks == 0
        return nothing
    end

    # Launch kernel
    kernel(ctx, args...; threads, blocks)

    return nothing
end


function KI.KIKernel(::CUDABackend, f, args...; kwargs...)
    kern = eval(quote
        @cuda launch=false $(kwargs...) $(f)($(args...))
    end)
    KI.KIKernel{CUDABackend, typeof(kern)}(CUDABackend(), kern)
end

function (obj::KI.KIKernel{CUDABackend})(args...; numworkgroups=nothing, workgroupsize=nothing, kwargs...)
    threadsPerThreadgroup = isnothing(workgroupsize) ? 1 : workgroupsize
    threadgroupsPerGrid = isnothing(numworkgroups) ? 1 : numworkgroups

    obj.kern(args...; threads=threadsPerThreadgroup, blocks=threadgroupsPerGrid, kwargs...)
end


function KI.kernel_max_work_group_size(::CUDABackend, kikern::KI.KIKernel{<:CUDABackend}; max_work_items::Int=typemax(Int))::Int
    Int(min(kikern.kern.pipeline.maxTotalThreadsPerThreadgroup, max_work_items))
end
function KI.max_work_group_size(::CUDABackend)::Int
    Int(attribute(device(), DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))
end
function KI.multiprocessor_count(::CUDABackend)::Int
    Int(attribute(device(), DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
end

## indexing

## COV_EXCL_START
@device_override @inline function KI.get_local_id()
    return (; x = Int(threadIdx().x), y = Int(threadIdx().y), z = Int(threadIdx().z))
end


@device_override @inline function KI.get_group_id()
    return (; x = Int(blockIdx().x), y = Int(blockIdx().y), z = Int(blockIdx().z))
end

@device_override @inline function KI.get_global_id()
    return (; x = Int(blockDim().x), y = Int(blockDim().y), z = Int(blockDim().z))
end

@device_override @inline function KI.get_local_size()
    return (; x = Int((blockDim().x-1)*blockDim().x + threadIdx().x), y = Int((blockDim().y-1)*blockDim().y + threadIdx().y), z = Int((blockDim().z-1)*blockDim().z + threadIdx().z))
end

@device_override @inline function KI.get_num_grouups()
    return (; x = Int(gridDim().x), y = Int(gridDim().y), z = Int(gridDim().z))
end

@device_override @inline function KI.get_global_size()
    return (; x = Int(blockDim().x * gridDim().x), y = Int(blockDim().y * gridDim().y), z = Int(lockDim().z * gridDim().z))
end

@device_override @inline function KI.__validindex(ctx)
    if KA.__dynamic_checkbounds(ctx)
        I = @inbounds KA.expand(KA.__iterspace(ctx), blockIdx().x, threadIdx().x)
        return I in KA.__ndrange(ctx)
    else
        return true
    end
end

## shared and scratch memory

# @device_override @inline function KI.localmemory(::Type{T}, ::Val{Dims}, ::Val{Id}) where {T, Dims, Id}
@device_override @inline function KI.localmemory(::Type{T}, ::Val{Dims}) where {T, Dims}
    CuStaticSharedArray(T, Dims)
end

@device_override @inline function KA.Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    StaticArrays.MArray{KA.__size(Dims), T}(undef)
end

## synchronization and printing

@device_override @inline function KI.barrier()
    sync_threads()
end

@device_override @inline function KA.__print(args...)
    CUDA._cuprint(args...)
end

## COV_EXCL_STOP

## other

Adapt.adapt_storage(to::KA.ConstAdaptor, a::CuDeviceArray) = Base.Experimental.Const(a)

KA.argconvert(k::KA.Kernel{CUDABackend}, arg) = cudaconvert(arg)

function KA.priority!(::CUDABackend, prio::Symbol)
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
    CUDA.cuStreamGetFlags(old_stream, r_flags)
    flags = CUDA.CUstream_flags_enum(r_flags[])

    event = CuEvent(CUDA.EVENT_DISABLE_TIMING)
    record(event, old_stream)

    @debug "Switching default stream" flags priority
    new_stream = CuStream(; flags, priority)
    CUDA.wait(event, new_stream)
    stream!(new_stream)
    return nothing
end

end
