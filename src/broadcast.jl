# broadcasting

using Base.Broadcast: BroadcastStyle, Broadcasted

struct CuArrayStyle{N} <: AbstractGPUArrayStyle{N} end
CuArrayStyle(::Val{N}) where N = CuArrayStyle{N}()
CuArrayStyle{M}(::Val{N}) where {N,M} = CuArrayStyle{N}()

BroadcastStyle(::Type{CuArray{T,N}}) where {T,N} = CuArrayStyle{N}()

Base.similar(bc::Broadcasted{CuArrayStyle{N}}, ::Type{T}) where {N,T} =
    similar(CuArray{T}, axes(bc))

Base.similar(bc::Broadcasted{CuArrayStyle{N}}, ::Type{T}, dims) where {N,T} =
    CuArray{T}(undef, dims)

# broadcasting type ctors isn't GPU compatible
Broadcast.broadcasted(::CuArrayStyle{N}, f::Type{T}, args...) where {N, T} =
    Broadcasted{CuArrayStyle{N}}((x...) -> T(x...), args, nothing)
