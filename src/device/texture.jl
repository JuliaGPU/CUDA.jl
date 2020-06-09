export CuDeviceTexture

"""
    CuDeviceTexture{T,N,C,NC}

`N`-dimensional device texture with `C` channels of elements of type `T`. This type is the
device-side counterpart of [`CuTexture`](@ref), and can be used to access textures using
regular indexing notation. If `NC` is true, indices used by these accesses should be
normalized, i.e., fall into the `[0,1)` domain.

Device-side texture objects cannot be created directly, but should be created host-side
using [`CuTexture`](@ref) and passed to the kernal as an argument.
"""
struct CuDeviceTexture{T,N,C,NC}
    dims::Dims{N}
    handle::CUtexObject
end

Base.convert(::Type{CUtexObject}, t::CuDeviceTexture) = t.handle


## array interface

Base.size(tm::CuDeviceTexture) = tm.dims

Base.eltype(tm::CuDeviceTexture{T,N,1}) where {T,N} = T
Base.eltype(tm::CuDeviceTexture{T,N,C}) where {T,N,C} = NTuple{C,T}

isnormalized(t::CuDeviceTexture{<:Any,<:Any,<:Any,NC}) where {NC} = NC


## low-level operations

# Source: NVVM IR specification 1.4

for dims in 1:3,
    (dispatch_rettyp, julia_rettyp, llvm_rettyp) in
        ((Signed, NTuple{4,UInt32}, :v4u32),
         (Unsigned, NTuple{4,Int32}, :v4s32),
         (AbstractFloat, NTuple{4,Float32},:v4f32))

    llvm_dim = "$(dims)d"
    julia_args = (:x, :y, :z)[1:dims]
    julia_sig = ntuple(_->Float32, dims)
    julia_params = ntuple(i->:($(julia_args[i])::AbstractFloat), dims)

    @eval tex(texObject::CuDeviceTexture{<:$dispatch_rettyp,$dims}, $(julia_params...)) =
        ccall($"llvm.nvvm.tex.unified.$llvm_dim.$llvm_rettyp.f32", llvmcall,
            $julia_rettyp, (CUtexObject, $(julia_sig...)), texObject, $(julia_args...))


    # integer indices (tex?Dfetch) requires non-normalized coordinates

    julia_sig = ntuple(_->Int32, dims)
    julia_params = ntuple(i->:($(julia_args[i])::Integer), dims)

    @eval tex(texObject::CuDeviceTexture{<:$dispatch_rettyp,$dims,<:Any,false}, $(julia_params...)) =
        ccall($"llvm.nvvm.tex.unified.$llvm_dim.$llvm_rettyp.s32", llvmcall,
            $julia_rettyp, (CUtexObject, $(julia_sig...)), texObject, $(julia_args...))
end


## indexing

@inline function Base.getindex(t::CuDeviceTexture{T,N,C}, idx::Vararg{<:Real,N}) where {T,N,C}
    vals = if isnormalized(t)
        # normalized coordinates range between 0 and 1, and can be used as-is
        tex(t, idx...)
    else
        # non-normalized coordinates should be adjusted for 1-based indexing
        tex(t, ntuple(i->idx[i]-1, N)...)
    end

    # unpack the values
    return unpack(NTuple{C,T}, vals)
end

# unpack single-channel texture fetches as values, tuples otherwise
@inline unpack(::Type{NTuple{1,T}}, vals::NTuple) where T = unpack(T, vals[1])
@inline unpack(::Type{NTuple{C,T}}, vals::NTuple) where {C,T} = ntuple(i->unpack(T, vals[i]), C)

@inline unpack(::Type{T}, val::T) where {T} = val
@inline unpack(::Type{T}, val::Integer) where {T <: Integer} = unsafe_trunc(T, val)
@inline unpack(::Type{Float16}, val::Float32) = convert(Float16, val)
