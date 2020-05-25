# GPUArrays.jl interface


#
# Device functionality
#

## device properties

GPUArrays.threads(dev::CuDevice) =
    attribute(dev, DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)


## execution

struct CuArrayBackend <: AbstractGPUBackend end

struct CuKernelContext <: AbstractKernelContext end

# use the CUDA occupancy API to determine a launch configuration
function GPUArrays.gpu_call(::CuArrayBackend, f, args, total_threads::Int;
                            name::Union{String,Nothing})
    function configurator(kernel)
        config = launch_configuration(kernel.fun)

        threads = Base.min(total_threads, config.threads)
        blocks = cld(total_threads, threads)

        return (threads=threads, blocks=blocks)
    end

    @cuda config=configurator name=name f(CuKernelContext(), args...)
end

function GPUArrays.gpu_call(::CuArrayBackend, f, args, threads::Int, blocks::Int;
                            name::Union{String,Nothing})
    @cuda threads=threads blocks=blocks name=name f(CuKernelContext(), args...)
end

GPUArrays.synchronize(A::CuArray) = synchronize()


## on-device

# indexing

GPUArrays.blockidx(ctx::CuKernelContext) = CUDA.blockIdx().x
GPUArrays.blockdim(ctx::CuKernelContext) = CUDA.blockDim().x
GPUArrays.threadidx(ctx::CuKernelContext) = CUDA.threadIdx().x
GPUArrays.griddim(ctx::CuKernelContext) = CUDA.gridDim().x

# memory

@inline function GPUArrays.LocalMemory(::CuKernelContext, ::Type{T}, ::Val{dims}, ::Val{id}
                                      ) where {T, dims, id}
    ptr = CUDA._shmem(Val(id), T, Val(prod(dims)))
    CuDeviceArray(dims, DevicePtr{T, CUDA.AS.Shared}(ptr))
end

# synchronization

@inline GPUArrays.synchronize_threads(::CuKernelContext) = CUDA.sync_threads()



#
# Host abstractions
#

GPUArrays.device(A::CuArray) = device(CuCurrentContext())

GPUArrays.backend(::Type{<:CuArray}) = CuArrayBackend()

GPUArrays.unsafe_reinterpret(::Type{T}, A::CuArray, size::NTuple{N, Integer}) where {T, N} =
  CuArray{T,N}(convert(CuPtr{T}, pointer(A)), size, A)
