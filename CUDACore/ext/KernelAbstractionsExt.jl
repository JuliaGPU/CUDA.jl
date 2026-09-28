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
function KA.mkcontext(kernel::KA.Kernel{CUDABackend}, _ndrange, iterspace, launch)
    KA.CompilerMetadata{KA.ndrange(kernel), KA.DynamicCheck}(_ndrange, iterspace; launch)
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

function (obj::KA.Kernel{CUDABackend})(args...; ndrange=nothing, workgroupsize=nothing)
    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, workgroupsize)
    # nothing to launch (or compile) for an empty ndrange
    any(iszero, size(KA.blocks(iterspace))) && return nothing

    # launch on an N-d grid, computing indices in 32 bits, if possible. this doesn't depend
    # on the tuned workgroup size, so the context (and thus the kernel) doesn't either.
    launch = KA.select_launch(obj, ndrange, workgroupsize, iterspace)
    if launch === KA.NDLaunch{Int32}()
        # the common case, specialized statically
        launch_kernel(obj, KA.NDLaunch{Int32}(), ndrange, workgroupsize, iterspace, args...)
    else
        launch_kernel(obj, launch, ndrange, workgroupsize, iterspace, args...)
    end
    return nothing
end

function launch_kernel(obj::KA.Kernel{CUDABackend}, launch, ndrange, workgroupsize,
                       iterspace, args::Vararg{Any,N}) where {N}
    backend = KA.backend(obj)

    # this might not be the final context, since we may tune the workgroupsize
    ctx = KA.mkcontext(obj, ndrange, iterspace, launch)

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
        items = prod(KA.NDIteration.extents(ndrange))
        config = CUDACore.launch_configuration(kernel.fun; max_threads=items)
        if backend.prefer_blocks
            # Prefer blocks over threads
            threads = min(items, config.threads)
            # XXX: Some kernels performs much better with all blocks active
            cu_blocks = max(cld(items, threads), config.blocks)
            threads = cld(items, cu_blocks)
        else
            threads = config.threads
        end

        workgroupsize = KA.launch_workgroupsize(backend, launch, threads, ndrange)
        iterspace, dynamic = KA.partition(obj, ndrange, workgroupsize)
        ctx = KA.mkcontext(obj, ndrange, iterspace, launch)
        call = CUDACore.rebind(call, ctx, 1)
    end

    if launch isa KA.NDLaunch
        blocks = pad3(size(KA.blocks(iterspace)))
        threads = pad3(size(KA.workitems(iterspace)))
    else
        blocks = length(KA.blocks(iterspace))
        threads = length(KA.workitems(iterspace))
    end
    CUDACore.kernel_launch(kernel, call; threads, blocks)

    return nothing
end

pad3(t::Tuple) = (t..., ntuple(_ -> 1, 3 - length(t))...)

## scratch memory

@device_override @inline function KA.Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    StaticArrays.MArray{KA.__size(Dims), T}(undef)
end

## other

Adapt.adapt_storage(to::KA.ConstAdaptor, a::CuDeviceArray) = Base.Experimental.Const(a)

KA.argconvert(k::KA.Kernel{CUDABackend}, arg) = cudaconvert(arg)

end
