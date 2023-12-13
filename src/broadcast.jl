# broadcasting

using Base.Broadcast: BroadcastStyle, Broadcasted

struct CuArrayStyle{N,B} <: AbstractGPUArrayStyle{N} end
CuArrayStyle{M,B}(::Val{N}) where {N,M,B} = CuArrayStyle{N,B}()

# identify the broadcast style of a (wrapped) CuArray
BroadcastStyle(::Type{<:CuArray{T,N,B}}) where {T,N,B} = CuArrayStyle{N,B}()
BroadcastStyle(W::Type{<:AnyCuArray{T,N}}) where {T,N} =
    CuArrayStyle{N, buftype(Adapt.unwrap_type(W))}()

# when we are dealing with different buffer styles, we cannot know
# which one is better, so use unified memory
BroadcastStyle(::CUDA.CuArrayStyle{N, B1},
               ::CUDA.CuArrayStyle{N, B2}) where {N,B1,B2} =
    CuArrayStyle{N, Mem.Unified}()

# allocation of output arrays
Base.similar(bc::Broadcasted{CuArrayStyle{N,B}}, ::Type{T}, dims) where {T,N,B} =
    similar(CuArray{T,length(dims),B}, dims)
