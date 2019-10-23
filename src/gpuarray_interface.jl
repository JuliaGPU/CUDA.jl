import GPUArrays

struct CuArrayBackend <: GPUArrays.GPUBackend end
GPUArrays.backend(::Type{<:CuArray}) = CuArrayBackend()


#Abstract GPU interface
struct CuKernelState end

@inline function GPUArrays.LocalMemory(::CuKernelState, ::Type{T}, ::Val{N}, ::Val{id}
                                      ) where {T, N, id}
    ptr = CUDAnative._shmem(Val(id), T, Val(prod(N)))
    CuDeviceArray(N, DevicePtr{T, CUDAnative.AS.Shared}(ptr))
end

GPUArrays.AbstractDeviceArray(A::CUDAnative.CuDeviceArray, shape) = CUDAnative.CuDeviceArray(shape, pointer(A))

@inline GPUArrays.synchronize_threads(::CuKernelState) = CUDAnative.sync_threads()

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
GPUArrays.is_gpu(dev::CUDAdrv.CuDevice) = true
GPUArrays.name(dev::CUDAdrv.CuDevice) = string("CU ", CUDAdrv.name(dev))
GPUArrays.threads(dev::CUDAdrv.CuDevice) =
    CUDAdrv.attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)

GPUArrays.blocks(dev::CUDAdrv.CuDevice) =
    (CUDAdrv.attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),
     CUDAdrv.attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
     CUDAdrv.attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z))

GPUArrays.free_global_memory(dev::CUDAdrv.CuDevice) = CUDAdrv.Mem.info()[1]
GPUArrays.global_memory(dev::CUDAdrv.CuDevice) = CUDAdrv.totalmem(dev)
GPUArrays.local_memory(dev::CUDAdrv.CuDevice) =
    CUDAdrv.attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY)

function GPUArrays._gpu_call(::CuArrayBackend, f, A, args::Tuple,
                             blocks_threads::Tuple{T, T}) where {N, T <: NTuple{N, Integer}}
    blk, thr = blocks_threads
    @cuda blocks=blk threads=thr f(CuKernelState(), args...)
end

# Save reinterpret and reshape implementation use this in GPUArrays
function GPUArrays.unsafe_reinterpret(::Type{T}, A::CuArray,
                                      size::NTuple{N, Integer}) where {T, N}

  return CuArray{T,N}(convert(CuPtr{T}, A.ptr), size; base=A.base, own=A.own)
end

