# Contiguous on-device arrays (host side representation)

export
    CuArray


## construction

type CuArray{T,N} <: AbstractArray{T,N}
    devptr::DevicePtr{T}
    shape::NTuple{N,Int}
    len::Int

    ctx::CuContext

    function CuArray(shape::NTuple{N,Int})
        if !isbits(T)
            # non-isbits types results in an array with references to CPU objects
            throw(ArgumentError("CuArray with non-bit element type not supported"))
        elseif (sizeof(T) == 0)
            throw(ArgumentError("CuArray with zero-sized element types does not make sense"))
        end

        len = prod(shape)
        devptr = Mem.alloc(T, len)

        ctx = CuCurrentContext()
        obj = new(devptr, shape, len, ctx)
        track(ctx, obj)
        finalizer(obj, finalize)

        obj
    end

    function CuArray(shape::NTuple{N,Int}, devptr::DevicePtr{T})
        len = prod(shape)
        new(devptr, shape, len, CuContext(C_NULL))
    end
end

function finalize(a::CuArray)
    Mem.free(a.devptr)
    untrack(a.ctx, a)
end

(::Type{CuArray{T}}){T,N}(shape::NTuple{N,Int}) = CuArray{T,N}(shape)
(::Type{CuArray{T}}){T}(len::Int)               = CuArray{T,1}((len,))

function Base.:(==)(a::CuArray, b::CuArray)
    return a.ctx == b.ctx && pointer(a) == pointer(b)
end
Base.unsafe_convert{T}(::Type{DevicePtr{T}}, a::CuArray{T}) = a.devptr
Base.pointer(a::CuArray) = a.devptr

Base.similar{T}(a::CuArray{T,1})                    = CuArray{T}(length(a))
Base.similar{T}(a::CuArray{T,1}, S::Type)           = CuArray{S}(length(a))
Base.similar{T}(a::CuArray{T}, m::Int)              = CuArray{T}(m)
Base.similar{N}(a::CuArray, T::Type, dims::Dims{N}) = CuArray{T,N}(dims)
Base.similar{T,N}(a::CuArray{T}, dims::Dims{N})     = CuArray{T,N}(dims)


## array interface

Base.length(g::CuArray) = g.len
Base.size(g::CuArray) = g.shape


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
