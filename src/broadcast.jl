# broadcasting

using Base.Broadcast: BroadcastStyle, Broadcasted

struct CuArrayStyle{N,B} <: AbstractGPUArrayStyle{N} end
CuArrayStyle{M,B}(::Val{N}) where {N,M,B} = CuArrayStyle{N,B}()

# identify the broadcast style of a (wrapped) CuArray
BroadcastStyle(::Type{<:CuArray{T,N,B}}) where {T,N,B} = CuArrayStyle{N,B}()
BroadcastStyle(W::Type{<:AnyCuArray{T,N}}) where {T,N} =
    CuArrayStyle{N, memory_type(Adapt.unwrap_type(W))}()

# when we are dealing with different memory types, we cannot know
# which one is better, so use unified memory
BroadcastStyle(::CUDA.CuArrayStyle{N1, B1},
               ::CUDA.CuArrayStyle{N2, B2}) where {N1,N2,B1,B2} =
    CuArrayStyle{max(N1,N2), UnifiedMemory}()

# resolve ambiguity: different N, same memory type
BroadcastStyle(::CUDA.CuArrayStyle{N1, B},
               ::CUDA.CuArrayStyle{N2, B}) where {N1,N2,B} =
    CuArrayStyle{max(N1,N2), B}()

# allocation of output arrays
Base.similar(bc::Broadcasted{CuArrayStyle{N,B}}, ::Type{T}, dims) where {T,N,B} =
    similar(CuArray{T,length(dims),B}, dims)

# Base.Broadcast can't handle Int32 axes
# XXX: not using a quirk, as constprop/irinterpret is crucial here
# XXX: 1.11 uses to_index i nstead of CartesianIndex
Base.@propagate_inbounds Broadcast.newindex(arg::AnyCuDeviceArray, I::CartesianIndex) = CartesianIndex(_newindex(axes(arg), I.I))
Base.@propagate_inbounds Broadcast.newindex(arg::AnyCuDeviceArray, I::Integer) = CartesianIndex(_newindex(axes(arg), (I,)))
Base.@propagate_inbounds _newindex(ax::Tuple, I::Tuple) = # XXX: upstream this?
  (ifelse(length(ax[1]) == 1, promote(ax[1][1], I[1])...), _newindex(Base.tail(ax), Base.tail(I))...)
Base.@propagate_inbounds _newindex(ax::Tuple{}, I::Tuple) = ()
Base.@propagate_inbounds _newindex(ax::Tuple, I::Tuple{}) = (ax[1][1], _newindex(Base.tail(ax), ())...)
Base.@propagate_inbounds _newindex(ax::Tuple{}, I::Tuple{}) = ()
