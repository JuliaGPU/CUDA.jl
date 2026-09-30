module CUDAKernels

using ..CUDACore
using ..CUDACore: @device_override, default_memory, UnifiedMemory, GPUArrays

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

Adapt.adapt_storage(::CUDABackend, a::AbstractArray) = Adapt.adapt(CuArray, a)
Adapt.adapt_storage(::CUDABackend, a::Union{CuArray,GPUArrays.AbstractGPUSparseArray}) = a

## memory operations

# dense arrays, and contiguous views of them
const ContiguousArray{T} = Union{DenseArray{T}, Base.FastContiguousSubArray{T, <:Any, <:DenseArray}}
on_device(A::ContiguousArray) = parent(A) isa CuArray

function KI.copyto!(::CUDABackend, A::ContiguousArray{T}, B::ContiguousArray{T}) where {T}
    length(A) == length(B) ||
        throw(ArgumentError("Arrays must have the same length, got $(length(A)) and $(length(B))"))
    if isbitstype(T) && (on_device(A) || on_device(B))
        GC.@preserve A B CUDACore.with_managed_arrays(A, B) do
            unsafe_copyto!(pointer(A), pointer(B), length(A), async=true)
        end
    else
        # host-to-host copies, and bits unions, whose type tags are stored separately
        copyto!(A, B)
    end
    return A
end
KI.copyto!(::CUDABackend, A, B) =
    throw(ArgumentError("KernelInterface.copyto! only supports contiguous arrays of the same element type, got $(typeof(A)) and $(typeof(B))"))

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

# a compiled kernel, and the callable it was compiled from. that one is converted again at
# every launch, like the arguments, which keeps the arrays it captures alive and registers
# their managed memory.
struct CompiledKernel{K,F}
    kernel::K
    f::F
end

function KI.kernel_function(backend::CUDABackend, f::F, tt::TT=Tuple{}; name=nothing, kwargs...) where {F,TT}
    kernel = cufunction(cudaconvert(f), tt; name, backend.always_inline, backend.fastmath, kwargs...)
    KI.Kernel(backend, CompiledKernel(kernel, f))
end

# passes the arguments on as a tuple, like calling the `HostKernel` does
function KI.launch(obj::KI.Kernel{CUDABackend}, groups::Dims{3}, items::Dims{3},
                   args::Tuple; kwargs...)
    # KernelInterface has validated the launch geometry
    if haskey(kwargs, :threads) || haskey(kwargs, :blocks)
        throw(ArgumentError("KernelInterface kernels take `numgroups`, `workgroupsize` or `ndrange`, not `threads` or `blocks`"))
    end
    call = CUDACore.kernel_call(CUDACore.LLVMBackend(), obj.kern.f, args)
    CUDACore.kernel_launch(obj.kern.kernel, call; threads=items, blocks=groups, kwargs...)
    return
end

KI.max_work_group_size(kernel::KI.Kernel{CUDABackend})::Int = CUDACore.maxthreads(kernel.kern.kernel)
function KI.launch_configuration(kernel::KI.Kernel{CUDABackend}; nitems::Union{Integer,Nothing}=nothing,
                                 max_work_group_size::Integer=typemax(Int))
    max_threads = min(max_work_group_size, something(nitems, typemax(Int)), typemax(Int32))
    config = launch_configuration(kernel.kern.kernel.fun; max_threads)
    threads = Int(config.threads)
    if kernel.backend.prefer_blocks && nitems !== nothing
        # prefer blocks over threads: at least as many blocks as the occupancy API suggests
        # XXX: some kernels perform much better with all blocks active
        blocks = max(cld(nitems, threads), Int(config.blocks))
        threads = cld(nitems, blocks)
    end
    return (; workgroupsize=threads)
end
# these limits are the same for every supported device, so don't query them on every launch
KI.max_work_group_size(::CUDABackend)::Int = 1024
KI.max_work_group_dims(::CUDABackend)::NTuple{3, Int} = (1024, 1024, 64)
KI.max_num_groups(::CUDABackend)::NTuple{3, Int} = (Int(typemax(Int32)), 65535, 65535)
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

## shared and scratch memory

@device_override @inline function KI.localmemory(::Type{T}, ::Val{Dims}) where {T, Dims}
    CuStaticSharedArray(T, Dims)
end

## synchronization and printing

@device_override @inline function KI.barrier()
    sync_threads()
end

@device_override @inline function KI._print(args...)
    CUDACore._cuprint(args...)
end

## other

function KI.priority!(::CUDABackend, prio::Symbol)
    if !(prio in (:high, :normal, :low))
        error("priority must be one of :high, :normal, :low")
    end
    CUDACore.priority!(prio)
    return nothing
end

end
