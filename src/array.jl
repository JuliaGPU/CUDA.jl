# Contiguous on-device arrays (host side representation)

export
    CuArray, CuVector, CuMatrix


## construction

"""
    CuArray{T}(dims)
    CuArray{T,N}(dims)

Construct an uninitialized `N`-dimensional dense CUDA array with element type `T`, where `N`
is determined from the length or number of `dims`. `dims` may be a tuple or a series of
integer arguments corresponding to the lengths in each dimension. If the rank `N` is
supplied explicitly as in `Array{T,N}(dims)`, then it must match the length or number of
`dims`.

Type aliases `CuVector` and `CuMatrix` are available for respectively 1 and 2-dimensional
data.
"""
CuArray

mutable struct CuArray{T,N} <: AbstractArray{T,N}
    buf::Mem.Buffer
    shape::NTuple{N,Int}

    # inner constructors (exact types, ie. Int not <:Integer)
    function CuArray{T,N}(shape::NTuple{N,Int}) where {T,N}
        check_type(CuArray, T)

        len = prod(shape)
        buf = Mem.alloc(len*sizeof(T))
        retain(buf)

        obj = new{T,N}(buf, shape)
        @compat finalizer(unsafe_free!, obj)
        return obj
    end
    function CuArray{T,N}(shape::NTuple{N,Int}, buf::Buffer) where {T,N}
        check_type(CuArray, T)

        retain(buf)

        obj = new{T, N}(buf, shape)
        @compat finalizer(unsafe_free!, obj)
        return obj
    end
end

function check_type(::Type{CuArray}, ::Type{T}) where {T}
    if !isbits(T)
        # non-isbits types results in an array with references to CPU objects
        throw(ArgumentError("CuArray with non-bit element type not supported"))
    elseif (sizeof(T) == 0)
        throw(ArgumentError("CuArray with zero-sized element types does not make sense"))
    end
end

const CuVector{T} = CuArray{T,1}
const CuMatrix{T} = CuArray{T,2}

function unsafe_free!(a::CuArray)
    if !release(a.buf)
        @trace("Skipping finalizer for CuArray object at $(Base.pointer_from_objref(a))) because pointer is held by another object")
    elseif !isvalid(a.buf.ctx)
        @trace("Skipping finalizer for CuArray object at $(Base.pointer_from_objref(a))) because context is no longer valid")
    else
        @trace("Finalizing CuArray object at $(Base.pointer_from_objref(a))")
        Mem.free(a.buf)
    end
end


## construction

# outer constructors, partially parameterized
CuArray{T}(dims::NTuple{N,I}) where {T,N,I<:Integer}   = CuArray{T,N}(dims)
CuArray{T}(dims::Vararg{I,N}) where {T,N,I<:Integer}   = CuArray{T,N}(dims)

# outer constructors, fully parameterized
CuArray{T,N}(dims::NTuple{N,I}) where {T,N,I<:Integer} = CuArray{T,N}(Int.(dims))
CuArray{T,N}(dims::Vararg{I,N}) where {T,N,I<:Integer} = CuArray{T,N}(Int.(dims))

Base.similar(a::CuVector{T}) where {T}                     = CuArray{T}(length(a))
Base.similar(a::CuVector{T}, S::Type) where {T}            = CuArray{S}(length(a))
Base.similar(a::CuArray{T}, m::Int) where {T}              = CuArray{T}(m)
Base.similar(a::CuArray, T::Type, dims::Dims{N}) where {N} = CuArray{T,N}(dims)
Base.similar(a::CuArray{T}, dims::Dims{N}) where {T,N}     = CuArray{T,N}(dims)


## getters

Base.size(a::CuArray) = a.shape
Base.length(a::CuArray) = prod(a.shape)
Base.sizeof(a::CuArray{T}) where {T} = Base.elsize(a) * length(a)


## conversions

Base.cconvert(::Type{Ptr{T}}, a::CuArray{T}) where {T} = a.buf


## comparisons

Base.:(==)(a::CuArray, b::CuArray) = a.buf == b.buf
Base.hash(a::CuArray, h::UInt) = hash(a.buf, h)

# override the Base isequal, which compares values
Base.isequal(a::CuArray, b::CuArray) = a == b


## other

Base.showarray(io::IO, a::CuArray, repr::Bool = true; kwargs...) =
    Base.showarray(io, Array(a), repr; kwargs...)


## memory management

"""
    copy!{T}(dst::CuArray{T}, src::Array{T})

Copy an array from a host array `src` to a device array `dst` in place. Both arrays should
have an equal length.
"""
function Base.copy!(dst::CuArray{T}, src::Array{T}) where T
    if length(dst) != length(src)
        throw(ArgumentError("Inconsistent array length."))
    end
    Mem.upload!(dst.buf, pointer(src), length(src) * sizeof(T))
    return dst
end

"""
    copy!{T}(dst::Array{T}, src::CuArray{T})

Copy an array from a device array `src` to a host array `dst` in place. Both arrays should
have an equal length.
"""
function Base.copy!(dst::Array{T}, src::CuArray{T}) where T
    if length(dst) != length(src)
        throw(ArgumentError("Inconsistent array length."))
    end
    Mem.download!(pointer(dst), src.buf, length(src) * sizeof(T))
    return dst
end

"""
    copy!{T}(dst::CuArray{T}, src::CuArray{T})

Copy an array from a device array `src` to a device array `dst` in place. Both arrays should
have an equal length.
"""
function Base.copy!(dst::CuArray{T}, src::CuArray{T}) where T
    if length(dst) != length(src)
        throw(ArgumentError("Inconsistent array length."))
    end
    Mem.transfer!(dst.buf, src.buf, length(src) * sizeof(T))
    return dst
end


### convenience functions

"""
    CuArray{T}(src::Array{T})

Transfer a host array `src` to device, returning a [`CuArray`](@ref).
"""
CuArray(src::Array{T,N}) where {T,N} = copy!(CuArray{T,N}(size(src)), src)

"""
    Array{T}(src::CuArray{T})

Transfer a device array `src` to host, returning an `Array`.
"""
Base.Array(src::CuArray{T,N}) where {T,N} = copy!(Array{T,N}(uninitialized, size(src)), src)
