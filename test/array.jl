# a lightweight CUDA array type for testing purposes

## ctor & finalizer

mutable struct CuTestArray{T,N}
    buf::Mem.DeviceBuffer
    shape::NTuple{N,Int}
    function CuTestArray{T,N}(shape::NTuple{N,Int}) where {T,N}
        len = prod(shape)
        buf = Mem.alloc(Mem.Device, len*sizeof(T))

        obj = new{T,N}(buf, shape)
        finalizer(unsafe_free!, obj)
        return obj
    end
end

function unsafe_free!(a::CuTestArray)
    CUDAdrv.isvalid(a.buf.ctx) && Mem.free(a.buf)
end

Base.cconvert(::Type{<:CuPtr}, x::CuTestArray) = x.buf

Base.length(xs::CuTestArray) = prod(xs.shape)


## memory copy operations

function CuTestArray(src::Array{T,N}) where {T,N}
    dst = CuTestArray{T,N}(size(src))
    Mem.copy!(dst.buf, pointer(src), length(src) * sizeof(T))
    return dst
end

function Base.Array(src::CuTestArray{T,N}) where {T,N}
    dst = Array{T,N}(undef, src.shape)
    Mem.copy!(pointer(dst), src.buf, length(src) * sizeof(T))
    return dst
end