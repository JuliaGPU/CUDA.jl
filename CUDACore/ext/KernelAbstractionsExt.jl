module KernelAbstractionsExt

using CUDACore
using CUDACore: @device_override, GPUArrays

import KernelAbstractions as KA

import StaticArrays

import Adapt

# TODO: Move AbstractGPUSparseArray stuff out
Adapt.adapt_storage(::KA.CPU, a::Union{CuArray,GPUArrays.AbstractGPUSparseArray}) = Adapt.adapt(Array, a)

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

# distribute `threads` over the dimensions of `ndrange`, filling the first ones first.
# written recursively, because a closure that updates the running total would box it.
threads_to_workgroupsize(threads, ndrange::Tuple) = _threads_to_workgroupsize(threads, 1, ndrange)
_threads_to_workgroupsize(threads, total, ::Tuple{}) = ()
function _threads_to_workgroupsize(threads, total, ndrange::Tuple)
    x = min(div(threads, total), first(ndrange))
    return (x, _threads_to_workgroupsize(threads, total * x, Base.tail(ndrange))...)
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

    call = CUDACore.kernel_call(obj.f, (ctx, args...))
    kernel = CUDACore.kernel_compile(call; always_inline=backend.always_inline,
                                     fastmath=backend.fastmath, maxthreads)

    # figure out the optimal workgroupsize automatically
    if KA.workgroupsize(obj) <: KA.DynamicSize && workgroupsize === nothing
        config = CUDACore.launch_configuration(kernel.fun; max_threads=prod(ndrange))
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
        call = CUDACore.rebind(call, ctx, 1)
    end

    blocks = length(KA.blocks(iterspace))
    threads = length(KA.workitems(iterspace))

    if blocks == 0
        return nothing
    end

    # Launch kernel
    CUDACore.kernel_launch(kernel, call; threads, blocks)

    return nothing
end

## indexing

@device_override @inline function KA.__validindex(ctx)
    if KA.__dynamic_checkbounds(ctx)
        I = @inbounds KA.expand(KA.__iterspace(ctx), blockIdx().x, threadIdx().x)
        return I in KA.__ndrange(ctx)
    else
        return true
    end
end

## scratch memory

@device_override @inline function KA.Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    StaticArrays.MArray{KA.__size(Dims), T}(undef)
end

## other

Adapt.adapt_storage(to::KA.ConstAdaptor, a::CuDeviceArray) = Base.Experimental.Const(a)

KA.argconvert(k::KA.Kernel{CUDABackend}, arg) = cudaconvert(arg)

end
