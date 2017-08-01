Base.Broadcast._containertype(::Type{<:CuArray}) = CuArray

using Base.Cartesian
using Base.Broadcast: newindex, _broadcast_getindex

@generated function _broadcast!(f, C::AbstractArray, keeps::K, Idefaults::ID, A::AT, Bs::BT, ::Type{Val{N}}) where {K,ID,AT,BT,N}
    nargs = N + 1
    quote
        $(Expr(:meta, :inline))
        A_1 = A
        @nexprs $N i-> A_{i+1} = Bs[i]
        @nexprs $nargs i -> keep_i = keeps[i]
        @nexprs $nargs i -> Idefault_i = Idefaults[i]
        for i = 1:length(C)
            I = CartesianIndex(ind2sub(C, i))
            @nexprs $nargs i -> I_i = newindex(I, keep_i, Idefault_i)
            @nexprs $nargs i -> @inbounds val_i = _broadcast_getindex(A_i, I_i)
            result = @ncall $nargs f val
            @inbounds C[I] = result
        end
    end
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
    _broadcast!(f, C, keeps, Idefaults, A, Bs, Val{N})
    return C
end

@inline function Base.Broadcast.broadcast_c!(f, ::Type{CuArray}, ::Type, C, A, Bs::Vararg{Any,N}) where N
    shape = indices(C)
    @boundscheck check_broadcast_indices(shape, A, Bs...)
    keeps, Idefaults = map_newindexer(shape, A, Bs)
    _broadcast!(f, C, keeps, Idefaults, A, Bs, Val{N})
    return C
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
