module CUDAKernels

import KernelAbstractions
import CUDA

struct CUDADevice{PreferBlocks, AlwaysInline} <: GPU end
CUDADevice(;prefer_blocks=false, always_inline=false) = CUDADevice{prefer_blocks, always_inline}()
CUDADevice{PreferBlocks}() where PreferBlocks = CUDADevice{PreferBlocks, false}()

export CUDADevice

# Import through parent
import KernelAbstractions: StaticArrays, Adapt
import .StaticArrays: MArray
import KernelAbstractions: synchronize

KernelAbstractions.get_device(::CUDA.CuArray) = CUDADevice()
KernelAbstractions.get_device(::CUDA.CUSPARSE.AbstractCuSparseArray) = CUDADevice()

synchronize(::CUDADevice) = CUDA.synchronize()

###
# async_copy
###
# - IdDict does not free the memory
# - WeakRef dict does not unique the key by objectid
const __pinned_memory = Dict{UInt64, WeakRef}()

function __pin!(a)
    # use pointer instead of objectid?
    oid = objectid(a)
    if haskey(__pinned_memory, oid) && __pinned_memory[oid].value !== nothing
        return nothing
    end
    ad = CUDA.Mem.register(CUDA.Mem.Host, pointer(a), sizeof(a))
    finalizer(_ -> CUDA.Mem.unregister(ad), a)
    __pinned_memory[oid] = WeakRef(a)
    return nothing
end

function KernelAbstractions.async_copy!(::CUDADevice, A, B, progress=yield)
    A isa Array && __pin!(A)
    B isa Array && __pin!(B)

    GC.@preserve A B begin
        destptr = pointer(A)
        srcptr  = pointer(B)
        N       = length(A)
        unsafe_copyto!(destptr, srcptr, N, async=true)
    end
end

import KernelAbstractions: Kernel, StaticSize, DynamicSize, partition, blocks, workitems, launch_config

###
# Kernel launch
###
function launch_config(kernel::Kernel{<:CUDADevice}, ndrange, workgroupsize)
    if ndrange isa Integer
        ndrange = (ndrange,)
    end
    if workgroupsize isa Integer
        workgroupsize = (workgroupsize, )
    end

    # partition checked that the ndrange's agreed
    if KernelAbstractions.ndrange(kernel) <: StaticSize
        ndrange = nothing
    end

    iterspace, dynamic = if KernelAbstractions.workgroupsize(kernel) <: DynamicSize &&
        workgroupsize === nothing
        # use ndrange as preliminary workgroupsize for autotuning
        partition(kernel, ndrange, ndrange)
    else
        partition(kernel, ndrange, workgroupsize)
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

function (obj::Kernel{CUDADevice{PreferBlocks,AlwaysInline}})(args...; ndrange=nothing, workgroupsize=nothing, progress=yield) where {PreferBlocks,AlwaysInline}

    ndrange, workgroupsize, iterspace, dynamic = launch_config(obj, ndrange, workgroupsize)
    # this might not be the final context, since we may tune the workgroupsize
    ctx = mkcontext(obj, ndrange, iterspace)

    # If the kernel is statically sized we can tell the compiler about that
    if KernelAbstractions.workgroupsize(obj) <: StaticSize
        maxthreads = prod(KernelAbstractions.get(KernelAbstractions.workgroupsize(obj)))
    else
        maxthreads = nothing
    end

    kernel = CUDA.@cuda launch=false always_inline=AlwaysInline maxthreads=maxthreads obj.f(ctx, args...)

    # figure out the optimal workgroupsize automatically
    if KernelAbstractions.workgroupsize(obj) <: DynamicSize && workgroupsize === nothing
        config = CUDA.launch_configuration(kernel.fun; max_threads=prod(ndrange))
        if PreferBlocks
            # Prefer blocks over threads
            threads = min(prod(ndrange), config.threads)
            # XXX: Some kernels performs much better with all blocks active
            cu_blocks = max(cld(prod(ndrange), threads), config.blocks)
            threads = cld(prod(ndrange), cu_blocks)
        else
            threads = config.threads
        end

        workgroupsize = threads_to_workgroupsize(threads, ndrange)
        iterspace, dynamic = partition(obj, ndrange, workgroupsize)
        ctx = mkcontext(obj, ndrange, iterspace)
    end

    nblocks = length(blocks(iterspace))
    threads = length(workitems(iterspace))

    if nblocks == 0
        return nothing 
    end

    # Launch kernel
    kernel(ctx, args...; threads=threads, blocks=nblocks)

    return nothing
end

import CUDA: @device_override

import KernelAbstractions: CompilerMetadata, DynamicCheck, LinearIndices
import KernelAbstractions: __index_Local_Linear, __index_Group_Linear, __index_Global_Linear, __index_Local_Cartesian, __index_Group_Cartesian, __index_Global_Cartesian, __validindex, __print
import KernelAbstractions: mkcontext, expand, __iterspace, __ndrange, __dynamic_checkbounds

function mkcontext(kernel::Kernel{<:CUDADevice}, _ndrange, iterspace)
    CompilerMetadata{KernelAbstractions.ndrange(kernel), DynamicCheck}(_ndrange, iterspace)
end

@device_override @inline function __index_Local_Linear(ctx)
    return CUDA.threadIdx().x
end

@device_override @inline function __index_Group_Linear(ctx)
    return CUDA.blockIdx().x
end

@device_override @inline function __index_Global_Linear(ctx)
    I =  @inbounds expand(__iterspace(ctx), CUDA.blockIdx().x, CUDA.threadIdx().x)
    # TODO: This is unfortunate, can we get the linear index cheaper
    @inbounds LinearIndices(__ndrange(ctx))[I]
end

@device_override @inline function __index_Local_Cartesian(ctx)
    @inbounds workitems(__iterspace(ctx))[CUDA.threadIdx().x]
end

@device_override @inline function __index_Group_Cartesian(ctx)
    @inbounds blocks(__iterspace(ctx))[CUDA.blockIdx().x]
end

@device_override @inline function __index_Global_Cartesian(ctx)
    return @inbounds expand(__iterspace(ctx), CUDA.blockIdx().x, CUDA.threadIdx().x)
end

@device_override @inline function __validindex(ctx)
    if __dynamic_checkbounds(ctx)
        I = @inbounds expand(__iterspace(ctx), CUDA.blockIdx().x, CUDA.threadIdx().x)
        return I in __ndrange(ctx)
    else
        return true
    end
end

import KernelAbstractions: groupsize, __groupsize, __workitems_iterspace, add_float_contract, sub_float_contract, mul_float_contract

@static if Base.isbindingresolved(CUDA, :emit_shmem) && Base.isdefined(CUDA, :emit_shmem)
    const emit_shmem = CUDA.emit_shmem
else
    const emit_shmem = CUDA._shmem
end

import KernelAbstractions: ConstAdaptor, SharedMemory, Scratchpad, __synchronize, __size

###
# GPU implementation of shared memory
###

@device_override @inline function SharedMemory(::Type{T}, ::Val{Dims}, ::Val{Id}) where {T, Dims, Id}
    CUDA.CuStaticSharedArray(T, Dims)
end

###
# GPU implementation of scratch memory
# - private memory for each workitem
###

@device_override @inline function Scratchpad(ctx, ::Type{T}, ::Val{Dims}) where {T, Dims}
    MArray{__size(Dims), T}(undef)
end

@device_override @inline function __synchronize()
    CUDA.sync_threads()
end

@device_override @inline function __print(args...)
    CUDA._cuprint(args...)
end

###
# GPU implementation of const memory
###

Adapt.adapt_storage(to::ConstAdaptor, a::CUDA.CuDeviceArray) = Base.Experimental.Const(a)

# Argument conversion
KernelAbstractions.argconvert(k::Kernel{<:CUDADevice}, arg) = CUDA.cudaconvert(arg)

end
