Base.Broadcast._containertype(::Type{<:CuArray}) = CuArray

using Base.Cartesian
using Base.Broadcast: newindex, _broadcast_getindex

@generated function broadcast_kernel(f, C::AbstractArray, keeps::K, Idefaults::ID, A::AT, Bs::BT) where {K,ID,AT,BT}
    N = length(Bs.parameters)+1
    quote
        A_1 = A
        @nexprs $(N-1) i-> A_{i+1} = Bs[i]
        @nexprs $N i -> keep_i = keeps[i]
        @nexprs $N i -> Idefault_i = Idefaults[i]

        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        I = CartesianIndex(ind2sub(C, i))
        @nexprs $N i -> I_i = newindex(I, keep_i, Idefault_i)
        @nexprs $N i -> @inbounds val_i = _broadcast_getindex(A_i, I_i)
        result = @ncall $N f val
        @inbounds C[I] = result
    end
end

@inline function _broadcast!(f, C::AbstractArray, keeps, Idefaults, A, Bs)
    @cuda (1, length(C)) broadcast_kernel(f, todevice(C), keeps, Idefaults, todevice(A), todevice.(Bs))
    return C
end

using Base.Broadcast: map_newindexer, _broadcast_eltype, broadcast_indices,
    check_broadcast_indices

Base.Broadcast.broadcast_indices(::Type{CuArray}, A::Ref) = ()
Base.Broadcast.broadcast_indices(::Type{CuArray}, A) = indices(A)

# TODO: computed eltype broadcast?
@inline function broadcast_t(f, T, shape, A, Bs::Vararg{Any,N}) where N
    isleaftype(T) || error("Broadcast output type $T is not concrete")
    C = similar(CuArray{T}, shape)
    keeps, Idefaults = map_newindexer(shape, A, Bs)
    _broadcast!(f, C, keeps, Idefaults, A, Bs)
end

# Called by Base broadcasting mechanisms (in place and out of place)

@inline function Base.Broadcast.broadcast_c!(f, ::Type{CuArray}, ::Type, C, A, Bs::Vararg{Any,N}) where N
    shape = indices(C)
    @boundscheck check_broadcast_indices(shape, A, Bs...)
    keeps, Idefaults = map_newindexer(shape, A, Bs)
    _broadcast!(f, C, keeps, Idefaults, A, Bs)
end

@inline function Base.Broadcast.broadcast_c(f, ::Type{CuArray}, A, Bs...)
    T = _broadcast_eltype(f, A, Bs...)
    shape = broadcast_indices(A, Bs...)
    iter = CartesianRange(shape)
    if isleaftype(T)
        return broadcast_t(f, T, shape, A, Bs...)
    end
    if isempty(iter)
        return similar(CuArray{T}, shape)
    end
    return broadcast_t(f, Any, shape, A, Bs...)
end
