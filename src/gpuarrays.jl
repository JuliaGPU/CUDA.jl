# GPUArrays.jl interface


#
# Device functionality
#


## execution

struct CuArrayBackend <: AbstractGPUBackend end

struct CuKernelContext <: AbstractKernelContext end

@inline function GPUArrays.launch_heuristic(::CuArrayBackend, f::F, args::Vararg{Any,N};
                                            elements::Int, elements_per_thread::Int) where {F,N}
    kernel = @cuda launch=false f(CuKernelContext(), args...)

    # launching many large blocks) lowers performance, as observed with broadcast, so cap
    # the block size if we don't have a grid-stride kernel (which would keep the grid small)
    if elements_per_thread > 1
        launch_configuration(kernel.fun)
    else
        launch_configuration(kernel.fun; max_threads=256)
    end
end

@inline function GPUArrays.gpu_call(::CuArrayBackend, f::F, args::TT, threads::Int,
                                    blocks::Int; name::Union{String,Nothing}) where {F,TT}
    @cuda threads blocks name f(CuKernelContext(), args...)
end


## on-device

# indexing

GPUArrays.blockidx(ctx::CuKernelContext) = blockIdx().x
GPUArrays.blockdim(ctx::CuKernelContext) = blockDim().x
GPUArrays.threadidx(ctx::CuKernelContext) = threadIdx().x
GPUArrays.griddim(ctx::CuKernelContext) = gridDim().x

# memory

@inline function GPUArrays.LocalMemory(::CuKernelContext, ::Type{T}, ::Val{dims}, ::Val{id}
                                      ) where {T, dims, id}
    ptr = CUDA._shmem(Val(id), T, Val(prod(dims)))
    ptr = reinterpret(LLVMPtr{T, AS.Shared}, ptr)
    CuDeviceArray{T,length(dims),AS.Shared}(ptr, dims)
end

# synchronization

@inline GPUArrays.synchronize_threads(::CuKernelContext) = sync_threads()



#
# Host abstractions
#

GPUArrays.backend(::Type{<:CuArray}) = CuArrayBackend()
