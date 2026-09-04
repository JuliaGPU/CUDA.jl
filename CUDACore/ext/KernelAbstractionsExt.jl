module KernelAbstractionsExt

using CUDACore
using CUDACore: @device_override, default_memory, UnifiedMemory, GPUArrays

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

function threads_to_workgroupsize(threads, ndrange)
    total = Ref(1)
    return map(ndrange) do n
        x = min(div(threads, total[]), n)
        total[] *= x
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

    call = CUDACore.kernel_call(obj.f, (ctx, args...))
    kernel = CUDACore.kernel_compile(call; always_inline=backend.always_inline, maxthreads)

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

## COV_EXCL_START
@device_override @inline function KA.__validindex(ctx)
    if KA.__dynamic_checkbounds(ctx)
        I = @inbounds KA.expand(KA.__iterspace(ctx), blockIdx().x, threadIdx().x)
        return I in KA.__ndrange(ctx)
    else
        return true
    end
end

## shared and scratch memory

@device_override @inline function KA.Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    StaticArrays.MArray{KA.__size(Dims), T}(undef)
end

## COV_EXCL_STOP

## other

Adapt.adapt_storage(to::KA.ConstAdaptor, a::CuDeviceArray) = Base.Experimental.Const(a)

KA.argconvert(k::KA.Kernel{CUDABackend}, arg) = cudaconvert(arg)

end
