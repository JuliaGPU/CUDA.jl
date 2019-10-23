# a lightweight CUDA array type for testing purposes

## ctor & finalizer

mutable struct CuTestArray{T,N}
    ptr::CuPtr{T}
    shape::NTuple{N,Int}
    function CuTestArray{T,N}(shape::NTuple{N,Int}) where {T,N}
        len = prod(shape)
        buf = Mem.alloc(Mem.Device, len*sizeof(T))
        ptr = convert(CuPtr{T}, buf)

        obj = new{T,N}(ptr, shape)
        finalizer(obj) do a
            CUDAdrv.isvalid(buf.ctx) && Mem.free(buf)
        end
        return obj
    end
    function CuTestArray{T,N}(ptr::CuPtr{T}, shape::NTuple{N,Int}) where {T,N}
        new{T,N}(ptr, shape)
    end
end

Base.pointer(xs::CuTestArray) = xs.ptr

Base.length(xs::CuTestArray) = prod(xs.shape)

Base.unsafe_convert(::Type{<:CuPtr}, x::CuTestArray) = pointer(x)


## memory copy operations

function CuTestArray(src::Array{T,N}) where {T,N}
    dst = CuTestArray{T,N}(size(src))
    unsafe_copyto!(pointer(dst), pointer(src), length(src))
    return dst
end

function Base.Array(src::CuTestArray{T,N}) where {T,N}
    dst = Array{T,N}(undef, src.shape)
    unsafe_copyto!(pointer(dst), pointer(src), length(src))
    return dst
end


## conversions

using Adapt
function Adapt.adapt_storage(::CUDAnative.Adaptor, a::CuTestArray{T,N}) where {T,N}
    ptr = pointer(a)
    devptr = CUDAnative.DevicePtr{T,AS.Global}(ptr)
    CuDeviceArray{T,N,AS.Global}(a.shape, devptr)
end
