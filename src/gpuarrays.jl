# GPUArrays.jl interface

import GPUArrays

struct CuArrayBackend <: AbstractGPUBackend end
GPUArrays.backend(::Type{<:CuArray}) = CuArrayBackend()

struct CuKernelState end

@inline function GPUArrays.LocalMemory(::CuKernelState, ::Type{T}, ::Val{dims}, ::Val{id}
                                      ) where {T, dims, id}
    ptr = CUDAnative._shmem(Val(id), T, Val(prod(dims)))
    CuDeviceArray(dims, DevicePtr{T, CUDAnative.AS.Shared}(ptr))
end

@inline GPUArrays.synchronize_threads(::CuKernelState) = CUDAnative.sync_threads()

## blas

GPUArrays.blas_module(::CuArray) = CuArrays.CUBLAS
GPUArrays.blasbuffer(x::CuArray) = x

"""
Blocks until all operations are finished on `A`
"""
GPUArrays.synchronize(A::CuArray) =
    CUDAdrv.synchronize()

for (i, sym) in enumerate((:x, :y, :z))
    for (f, fcu) in (
            (:blockidx, :blockIdx),
            (:blockdim, :blockDim),
            (:threadidx, :threadIdx),
            (:griddim, :gridDim)
        )
        fname = Symbol(string(f, '_', sym))
        cufun = Symbol(string(fcu, '_', sym))
        @eval GPUArrays.$fname(::CuKernelState) = CUDAnative.$cufun()
    end
end

# devices() = CUDAdrv.devices()
GPUArrays.device(A::CuArray) = CUDAdrv.device(CUDAdrv.CuCurrentContext())

# device properties
GPUArrays.threads(dev::CUDAdrv.CuDevice) =
    CUDAdrv.attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)

function GPUArrays._gpu_call(::CuArrayBackend, f, A, args::Tuple,
                             blocks_threads::Tuple{T, T}) where {N, T <: NTuple{N, Integer}}
    blk, thr = blocks_threads
    @cuda blocks=blk threads=thr f(CuKernelState(), args...)
end

# Save reinterpret and reshape implementation use this in GPUArrays
function GPUArrays.unsafe_reinterpret(::Type{T}, A::CuArray,
                                      size::NTuple{N, Integer}) where {T, N}

  return CuArray{T,N}(convert(CuPtr{T}, A.ptr), size, A)
end
