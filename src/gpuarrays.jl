# GPUArrays.jl interface


## host

function GPUArrays.unsafe_reinterpret(::Type{T}, A::CuArray,
                                      size::NTuple{N, Integer}) where {T, N}
  return CuArray{T,N}(convert(CuPtr{T}, A.ptr), size, A)
end

# execution

struct CuArrayBackend <: AbstractGPUBackend end
GPUArrays.backend(::Type{<:CuArray}) = CuArrayBackend()

struct CuKernelState end

function GPUArrays._gpu_call(::CuArrayBackend, f, A, args::Tuple,
                             blocks_threads::Tuple{T, T}) where {N, T <: NTuple{N, Integer}}
    blk, thr = blocks_threads
    @cuda blocks=blk threads=thr f(CuKernelState(), args...)
end

GPUArrays.synchronize(A::CuArray) = CUDAdrv.synchronize()

# device properties

GPUArrays.device(A::CuArray) = CUDAdrv.device(CUDAdrv.CuCurrentContext())

GPUArrays.threads(dev::CUDAdrv.CuDevice) =
    CUDAdrv.attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)

# linear algebra

GPUArrays.blas_module(::CuArray) = CuArrays.CUBLAS
GPUArrays.blasbuffer(x::CuArray) = x


## device

@inline function GPUArrays.LocalMemory(::CuKernelState, ::Type{T}, ::Val{dims}, ::Val{id}
                                      ) where {T, dims, id}
    ptr = CUDAnative._shmem(Val(id), T, Val(prod(dims)))
    CuDeviceArray(dims, DevicePtr{T, CUDAnative.AS.Shared}(ptr))
end

# indexing

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

# synchronization

@inline GPUArrays.synchronize_threads(::CuKernelState) = CUDAnative.sync_threads()
