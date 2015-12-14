# Contiguous on-device arrays

import Base: length, size, ndims, eltype, copy!

export
    CuArray, free, to_host, ndims


type CuArray{T,N}
    ptr::DevicePtr{Void}
    shape::NTuple{N,Int}
    len::Int
end

function CuArray(T::Type, len::Integer)
    if !isbits(T)
        throw(ArgumentError("CuArray  with non-bit element type not supported"))
    end
    n = Int(len)
    p = cualloc(T, n)
    CuArray{T,1}(p, (n,), n)
end

function CuArray{N}(T::Type, shape::NTuple{N,Int})
    if !isbits(T)
        throw(ArgumentError("CuArray  with non-bit element type not supported"))
    end
    n = prod(shape)
    p = cualloc(T, n)
    CuArray{T,N}(p, shape, n)
end

length(g::CuArray) = g.len
size(g::CuArray) = g.shape

ndims{T,N}(::CuArray{T,N}) = N
eltype{T,N}(::CuArray{T,N}) = T
eltype{T,N}(::Type{CuArray{T,N}}) = T

# TODO: can we avoid these duplicate definitions?
eltype{T}(::CuArray{T}) = T
eltype{T}(::Type{CuArray{T}}) = T

function size{T,N}(g::CuArray{T,N}, d::Integer)
    d >= 1 ? (d <= N ? g.shape[d] : 1) : error("Invalid index of dimension.")
end

function free(g::CuArray)
    if !isnull(g.ptr)
        free(g.ptr)
        g.ptr = DevicePtr{eltype(g.ptr)}()
    end
end

function copy!{T}(dst::Array{T}, src::CuArray{T})
    if length(dst) != length(src) 
        throw(ArgumentError("Inconsistent array length."))
    elseif !isbits(T)
        throw(ArgumentError("Copy of non-bits element type not supported"))
    end
    nbytes = length(src) * sizeof(T)
    @cucall(:cuMemcpyDtoH, (Ptr{Void}, Ptr{Void}, Csize_t),
                           pointer(dst), src.ptr.inner, nbytes)
    return dst
end

function copy!{T}(dst::CuArray{T}, src::Array{T})
    if length(dst) != length(src)
        throw(ArgumentError("Inconsistent array length."))
    elseif !isbits(T)
        throw(ArgumentError("Copy of non-bits element type not supported"))    
    end
    nbytes = length(src) * sizeof(T)
    @cucall(:cuMemcpyHtoD, (Ptr{Void}, Ptr{Void}, Csize_t),
                           dst.ptr.inner, pointer(src), nbytes)
    return dst
end

CuArray{T,N}(a::Array{T,N}) = copy!(CuArray(T, size(a)), a)
to_host{T}(g::CuArray{T}) = copy!(Array(T, size(g)), g)
