module CUDAKernels

import KernelAbstractions
import CUDA
import CUDA: @device_override
import UnsafeAtomicsLLVM
import GPUCompiler

struct CUDABackend <: KernelAbstractions.GPU
    prefer_blocks::Bool
    always_inline::Bool
end
CUDABackend(;prefer_blocks=false, always_inline=false) = CUDABackend(prefer_blocks, always_inline)

export CUDABackend

KernelAbstractions.unsafe_free!(x::CUDA.CuArray) = CUDA.unsafe_free!(x)
KernelAbstractions.allocate(::CUDABackend, ::Type{T}, dims::Tuple) where T = CUDA.CuArray{T}(undef, dims)
KernelAbstractions.zeros(::CUDABackend, ::Type{T}, dims::Tuple) where T = CUDA.zeros(T, dims)
KernelAbstractions.ones(::CUDABackend, ::Type{T}, dims::Tuple) where T = CUDA.ones(T, dims)

# Import through parent
import KernelAbstractions: StaticArrays, Adapt
import .StaticArrays: MArray

KernelAbstractions.get_backend(::CUDA.CuArray) = CUDABackend()
KernelAbstractions.get_backend(::CUDA.CUSPARSE.AbstractCuSparseArray) = CUDABackend()
KernelAbstractions.synchronize(::CUDABackend) = CUDA.synchronize()

Adapt.adapt_storage(::CUDABackend, a::Array) = Adapt.adapt(CUDA.CuArray, a)
Adapt.adapt_storage(::CUDABackend, a::CUDA.CuArray) = a
Adapt.adapt_storage(::KernelAbstractions.CPU, a::CUDA.CuArray) = convert(Array, a)

###
# copyto!
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

function KernelAbstractions.copyto!(::CUDABackend, A, B)
    A isa Array && __pin!(A)
    B isa Array && __pin!(B)

    GC.@preserve A B begin
        destptr = pointer(A)
        srcptr  = pointer(B)
        N       = length(A)
        unsafe_copyto!(destptr, srcptr, N, async=true)
    end
    return A
end

import KernelAbstractions: Kernel, StaticSize, DynamicSize, partition, blocks, workitems, launch_config

###
# Kernel launch
###
function launch_config(kernel::Kernel{CUDABackend}, ndrange, workgroupsize)
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

function (obj::Kernel{CUDABackend})(args...; ndrange=nothing, workgroupsize=nothing)
    backend = KernelAbstractions.backend(obj)

    ndrange, workgroupsize, iterspace, dynamic = launch_config(obj, ndrange, workgroupsize)
    # this might not be the final context, since we may tune the workgroupsize
    ctx = mkcontext(obj, ndrange, iterspace)

    # If the kernel is statically sized we can tell the compiler about that
    if KernelAbstractions.workgroupsize(obj) <: StaticSize
        maxthreads = prod(KernelAbstractions.get(KernelAbstractions.workgroupsize(obj)))
    else
        maxthreads = nothing
    end

    kernel = CUDA.@cuda launch=false always_inline=backend.always_inline maxthreads=maxthreads obj.f(ctx, args...)

    # figure out the optimal workgroupsize automatically
    if KernelAbstractions.workgroupsize(obj) <: DynamicSize && workgroupsize === nothing
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

import KernelAbstractions: CompilerMetadata, DynamicCheck, LinearIndices
import KernelAbstractions: __index_Local_Linear, __index_Group_Linear, __index_Global_Linear, __index_Local_Cartesian, __index_Group_Cartesian, __index_Global_Cartesian, __validindex, __print
import KernelAbstractions: mkcontext, expand, __iterspace, __ndrange, __dynamic_checkbounds

function mkcontext(kernel::Kernel{CUDABackend}, _ndrange, iterspace)
    CompilerMetadata{KernelAbstractions.ndrange(kernel), DynamicCheck}(_ndrange, iterspace)
end

@device_override function KernelAbstractions.isongpu()
    return true
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

import KernelAbstractions: groupsize, __groupsize, __workitems_iterspace
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
KernelAbstractions.argconvert(k::Kernel{CUDABackend}, arg) = CUDA.cudaconvert(arg)

##
# Priority
##

function KernelAbstractions.priority!(::CUDABackend, prio::Symbol)
    if !(prio in (:high, :normal, :low))
        error("priority must be one of :high, :normal, :low")
    end

    range = CUDA.priority_range()
    # 0:-1:-5
    # lower number is higher priority, default is 0
    # there is no "low"
    if prio === :high
        priority = last(range)
    elseif prio === :normal || prio === :low
        priority = first(range)
    end

    old_stream = CUDA.stream()
    r_flags = Ref{Cuint}()
    CUDA.cuStreamGetFlags(old_stream, r_flags)
    flags = CUDA.CUstream_flags_enum(r_flags[])

    event = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
    CUDA.record(event, old_stream)

    @debug "Switching default stream" flags priority
    new_stream = CUDA.CuStream(; flags, priority)
    CUDA.wait(event, new_stream)
    CUDA.stream!(new_stream)
    return nothing
end

end
