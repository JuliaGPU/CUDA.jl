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

function GPUArrays.launch_heuristic(::CuArrayBackend, f::F, args::Vararg{Any,N};
                                    maximize_blocksize=false) where {F,N}
    kernel_args = map(cudaconvert, args)
    kernel_tt = Tuple{CuKernelContext, map(Core.Typeof, kernel_args)...}
    kernel = cufunction(f, kernel_tt)
    if maximize_blocksize
        # some kernels benefit (algorithmically) from a large block size
        launch_configuration(kernel.fun)
    else
        # otherwise, using huge blocks often hurts performance: even though it maximizes
        # occupancy, we'd rather have a couple of blocks to switch between.
        launch_configuration(kernel.fun; max_threads=256)
    end
end

function GPUArrays.gpu_call(::CuArrayBackend, f, args, threads::Int, blocks::Int;
                            name::Union{String,Nothing})
    @cuda threads=threads blocks=blocks name=name f(CuKernelContext(), args...)
end


## on-device

# indexing

GPUArrays.blockidx(ctx::CuKernelContext) = CUDA.blockIdx().x
GPUArrays.blockdim(ctx::CuKernelContext) = CUDA.blockDim().x
GPUArrays.threadidx(ctx::CuKernelContext) = CUDA.threadIdx().x
GPUArrays.griddim(ctx::CuKernelContext) = CUDA.gridDim().x

# math

@inline GPUArrays.cos(ctx::CuKernelContext, x) = CUDA.cos(x)
@inline GPUArrays.sin(ctx::CuKernelContext, x) = CUDA.sin(x)
@inline GPUArrays.sqrt(ctx::CuKernelContext, x) = CUDA.sqrt(x)
@inline GPUArrays.log(ctx::CuKernelContext, x) = CUDA.log(x)

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
