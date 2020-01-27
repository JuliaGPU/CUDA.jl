# GPUArrays.jl interface


#
# Device functionality
#

## device properties

GPUArrays.threads(dev::CUDAdrv.CuDevice) =
    CUDAdrv.attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)


## execution

struct CuArrayBackend <: AbstractGPUBackend end

struct CuKernelContext <: AbstractKernelContext end

function GPUArrays.gpu_call(::CuArrayBackend, f, args...; threads::Int, blocks::Int)
    @cuda threads=threads blocks=blocks f(CuKernelContext(), args...)
end

GPUArrays.synchronize(A::CuArray) = CUDAdrv.synchronize()


## on-device

# indexing

GPUArrays.blockidx(ctx::CuKernelContext) = CUDAnative.blockIdx().x
GPUArrays.blockdim(ctx::CuKernelContext) = CUDAnative.blockDim().x
GPUArrays.threadidx(ctx::CuKernelContext) = CUDAnative.threadIdx().x
GPUArrays.griddim(ctx::CuKernelContext) = CUDAnative.gridDim().x

# memory

@inline function GPUArrays.LocalMemory(::CuKernelContext, ::Type{T}, ::Val{dims}, ::Val{id}
                                      ) where {T, dims, id}
    ptr = CUDAnative._shmem(Val(id), T, Val(prod(dims)))
    CuDeviceArray(dims, DevicePtr{T, CUDAnative.AS.Shared}(ptr))
end

# synchronization

@inline GPUArrays.synchronize_threads(::CuKernelContext) = CUDAnative.sync_threads()



#
# Host abstractions
#

GPUArrays.device(A::CuArray) = CUDAdrv.device(CUDAdrv.CuCurrentContext())

GPUArrays.backend(::Type{<:CuArray}) = CuArrayBackend()

GPUArrays.unsafe_reinterpret(::Type{T}, A::CuArray, size::NTuple{N, Integer}) where {T, N} =
  CuArray{T,N}(convert(CuPtr{T}, A.ptr), size, A)
