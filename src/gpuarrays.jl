# GPUArrays.jl interface


#
# Device functionality
#


## execution

struct CuArrayBackend <: AbstractGPUBackend end

struct CuKernelContext <: AbstractKernelContext end

@inline function GPUArrays.launch_heuristic(::CuArrayBackend, f::F, args::Vararg{Any,N};
                                            maximize_blocksize=false) where {F,N}
    kernel = @cuda launch=false f(CuKernelContext(), args...)
    if maximize_blocksize
        # some kernels benefit (algorithmically) from a large block size
        launch_configuration(kernel.fun)
    else
        # otherwise, using huge blocks often hurts performance: even though it maximizes
        # occupancy, we'd rather have a couple of blocks to switch between.
        launch_configuration(kernel.fun; max_threads=256)
    end
end

@inline function GPUArrays.gpu_call(::CuArrayBackend, f::F, args::TT, threads::Int,
                                    blocks::Int; name::Union{String,Nothing}) where {F,TT}
    @cuda threads=threads blocks=blocks name=name f(CuKernelContext(), args...)
end


## on-device

# indexing

GPUArrays.blockidx(ctx::CuKernelContext) = blockIdx().x
GPUArrays.blockdim(ctx::CuKernelContext) = blockDim().x
GPUArrays.threadidx(ctx::CuKernelContext) = threadIdx().x
GPUArrays.griddim(ctx::CuKernelContext) = gridDim().x

# math

@inline GPUArrays.cos(ctx::CuKernelContext, x) = cos(x)
@inline GPUArrays.sin(ctx::CuKernelContext, x) = sin(x)
@inline GPUArrays.sqrt(ctx::CuKernelContext, x) = sqrt(x)
@inline GPUArrays.log(ctx::CuKernelContext, x) = log(x)

# memory

@inline function GPUArrays.LocalMemory(::CuKernelContext, ::Type{T}, ::Val{dims}, ::Val{id}
                                      ) where {T, dims, id}
    ptr = CUDA._shmem(Val(id), T, Val(prod(dims)))
    CuDeviceArray(dims, reinterpret(LLVMPtr{T, AS.Shared}, ptr))
end

# synchronization

@inline GPUArrays.synchronize_threads(::CuKernelContext) = sync_threads()



#
# Host abstractions
#

GPUArrays.backend(::Type{<:CuArray}) = CuArrayBackend()
