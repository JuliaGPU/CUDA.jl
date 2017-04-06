# Contiguous on-device arrays (host side representation)

export
    CuArray


## construction

"""
    CuArray{T}(dims)
    CuArray{T,N}(dims)

Construct an uninitialized `N`-dimensional dense CUDA array with element type `T`, where `N`
is determined from the length or number of `dims`. `dims` may be a tuple or a series of
integer arguments corresponding to the lengths in each dimension. If the rank `N` is
supplied explicitly as in `Array{T,N}(dims)`, then it must match the length or number of
`dims`.
"""
CuArray

@compat type CuArray{T,N} <: AbstractArray{T,N}
    devptr::DevicePtr{T}
    shape::NTuple{N,Int}

    # inner constructors (exact types, ie. Int not <:Integer)
    function (::Type{CuArray{T,N}}){T,N}(shape::NTuple{N,Int})
        if !isbits(T)
            # non-isbits types results in an array with references to CPU objects
            throw(ArgumentError("CuArray with non-bit element type not supported"))
        elseif (sizeof(T) == 0)
            throw(ArgumentError("CuArray with zero-sized element types does not make sense"))
        end

        len = prod(shape)
        devptr = Mem.alloc(T, len)

        obj = new{T,N}(devptr, shape)
        finalizer(obj, free!)
        return obj
    end
    function (::Type{CuArray{T,N}}){T,N}(shape::NTuple{N,Int}, devptr::DevicePtr{T})
        # semi-hidden constructor, only called by unsafe_convert
        new{T,N}(devptr, shape)
    end
end

# outer constructors, partially parameterized
(::Type{CuArray{T}}){T,N,I<:Integer}(dims::NTuple{N,I})   = CuArray{T,N}(dims)
(::Type{CuArray{T}}){T,N,I<:Integer}(dims::Vararg{I,N})   = CuArray{T,N}(dims)

# outer constructors, fully parameterized
(::Type{CuArray{T,N}}){T,N,I<:Integer}(dims::NTuple{N,I}) = CuArray{T,N}(Int.(dims))
(::Type{CuArray{T,N}}){T,N,I<:Integer}(dims::Vararg{I,N}) = CuArray{T,N}(Int.(dims))

function free!(a::CuArray)
    if isvalid(a.devptr.ctx)
        @trace("Finalizing CuArray at $(Base.pointer_from_objref(a))")
        Mem.free(a.devptr)
    else
        @trace("Skipping finalizer for CuArray at $(Base.pointer_from_objref(a))) because context is no longer valid")
    end
end

Base.unsafe_convert{T}(::Type{DevicePtr{T}}, a::CuArray{T}) = a.devptr

Base.:(==)(a::CuArray, b::CuArray) = a.devptr == b.devptr
Base.hash(a::CuArray, h::UInt) = hash(a.devptr, h)

Base.pointer(a::CuArray) = a.devptr

# override the Base isequal, which compares values
Base.isequal(a::CuArray, b::CuArray) = a == b

Base.similar{T}(a::CuArray{T,1})                    = CuArray{T}(length(a))
Base.similar{T}(a::CuArray{T,1}, S::Type)           = CuArray{S}(length(a))
Base.similar{T}(a::CuArray{T}, m::Int)              = CuArray{T}(m)
Base.similar{N}(a::CuArray, T::Type, dims::Dims{N}) = CuArray{T,N}(dims)
Base.similar{T,N}(a::CuArray{T}, dims::Dims{N})     = CuArray{T,N}(dims)


## array interface

Base.size(g::CuArray) = g.shape
Base.length(g::CuArray) = prod(g.shape)

Base.showarray(io::IO, a::CuArray, repr::Bool = true; kwargs...) =
    Base.showarray(io, Array(a), repr; kwargs...)


## memory management

"Copy an array from host to device in place"
function Base.copy!{T}(dst::CuArray{T}, src::Array{T})
    if length(dst) != length(src)
        throw(ArgumentError("Inconsistent array length."))  
    end
    Mem.upload(dst.devptr, pointer(src), length(src) * sizeof(T))
    return dst
end

"Copy an array from device to host in place"
function Base.copy!{T}(dst::Array{T}, src::CuArray{T})
    if length(dst) != length(src)
        throw(ArgumentError("Inconsistent array length."))
    end
    Mem.download(pointer(dst), src.devptr, length(src) * sizeof(T))
    return dst
end

"Copy an array from device to device in place"
function Base.copy!{T}(dst::CuArray{T}, src::CuArray{T})
    if length(dst) != length(src)
        throw(ArgumentError("Inconsistent array length."))
    end
    Mem.transfer(dst.devptr, src.devptr, length(src) * sizeof(T))
    return dst
end


### convenience functions

"Transfer an array from host to device, returning a pointer on the device"
CuArray{T,N}(a::Array{T,N}) = copy!(CuArray{T}(size(a)), a)

"Transfer an array on the device to host"
Base.Array{T}(g::CuArray{T}) = copy!(Array{T}(size(g)), g)
