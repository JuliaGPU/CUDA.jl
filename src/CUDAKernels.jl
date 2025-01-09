module CUDAKernels

using ..CUDA
using ..CUDA: @device_override

import KernelAbstractions as KA

import StaticArrays

import Adapt

## back-end

export CUDABackend

struct CUDABackend <: KA.GPU
    prefer_blocks::Bool
    always_inline::Bool
end

CUDABackend(; prefer_blocks=false, always_inline=false) = CUDABackend(prefer_blocks, always_inline)

KA.allocate(::CUDABackend, ::Type{T}, dims::Tuple) where T = CuArray{T}(undef, dims)
KA.zeros(::CUDABackend, ::Type{T}, dims::Tuple) where T = CUDA.zeros(T, dims)
KA.ones(::CUDABackend, ::Type{T}, dims::Tuple) where T = CUDA.ones(T, dims)

KA.get_backend(::CuArray) = CUDABackend()
KA.get_backend(::CUSPARSE.AbstractCuSparseArray) = CUDABackend()
KA.synchronize(::CUDABackend) = synchronize()

Adapt.adapt_storage(::CUDABackend, a::Array) = Adapt.adapt(CuArray, a)
Adapt.adapt_storage(::CUDABackend, a::CuArray) = a
Adapt.adapt_storage(::KA.CPU, a::CuArray) = convert(Array, a)

## memory operations

function KA.copyto!(::CUDABackend, A, B)
    A isa Array && CUDA.pin(A)
    B isa Array && CUDA.pin(B)

    GC.@preserve A B begin
        destptr = pointer(A)
        srcptr  = pointer(B)
        N       = length(A)
        unsafe_copyto!(destptr, srcptr, N, async=true)
    end
    return A
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

## indexing

@device_override @inline function KA.__index_Local_Linear(ctx)
    return threadIdx().x
end


@device_override @inline function KA.__index_Group_Linear(ctx)
    return blockIdx().x
end

@device_override @inline function KA.__index_Global_Linear(ctx)
    # I =  @inbounds KA.expand(KA.__iterspace(ctx), blockIdx().x, threadIdx().x)
    return (blockIdx().x-1) * blockDim().x + threadIdx().x
    # TODO: This is unfortunate, can we get the linear index cheaper
    # @inbounds LinearIndices(KA.__ndrange(ctx))[I]
end

@device_override @inline function KA.__index_Local_Cartesian(ctx)
    @inbounds KA.workitems(KA.__iterspace(ctx))[threadIdx().x]
end

@device_override @inline function KA.__index_Group_Cartesian(ctx)
    @inbounds KA.blocks(KA.__iterspace(ctx))[blockIdx().x]
end

@device_override @inline function KA.__index_Global_Cartesian(ctx)
    return @inbounds KA.expand(KA.__iterspace(ctx), blockIdx().x, threadIdx().x)
end

@device_override @inline function KA.__validindex(ctx)
    if KA.__dynamic_checkbounds(ctx)
        I = @inbounds KA.expand(KA.__iterspace(ctx), blockIdx().x, threadIdx().x)
        return I in KA.__ndrange(ctx)
    else
        return true
    end
end

## shared and scratch memory

@device_override @inline function KA.SharedMemory(::Type{T}, ::Val{Dims}, ::Val{Id}) where {T, Dims, Id}
    CuStaticSharedArray(T, Dims)
end

@device_override @inline function KA.Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    StaticArrays.MArray{KA.__size(Dims), T}(undef)
end

## synchronization and printing

@device_override @inline function KA.__synchronize()
    sync_threads()
end

@device_override @inline function KA.__print(args...)
    CUDA._cuprint(args...)
end

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
