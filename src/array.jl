# a simple CUDA host array type for testing purposes.
# use CuArrays.jl for real applications!

mutable struct CuHostArray{T,N}
    ptr::CuPtr{T}
    dims::Dims{N}
end

Base.pointer(xs::CuHostArray) = xs.ptr

Base.size(xs::CuHostArray) = xs.dims

Base.length(xs::CuHostArray) = prod(xs.dims)

Base.unsafe_convert(::Type{<:CuPtr}, x::CuHostArray) = pointer(x)


## ctors & finalizer

function CuHostArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N}
    len = prod(dims)
    sz = len*sizeof(T)
    buf = try
        Mem.alloc(Mem.Device, sz)
    catch err
        (isa(err, CuError) && err.code == CUDAdrv.ERROR_OUT_OF_MEMORY) || rethrow()
        GC.gc(true)
        Mem.alloc(Mem.Device, sz)
    end
    ptr = convert(CuPtr{T}, buf)

    obj = CuHostArray{T,N}(ptr, dims)
    finalizer(obj) do a
        CUDAdrv.isvalid(buf.ctx) && Mem.free(buf)
    end
    return obj
end

# underspecified
CuHostArray{T,N}(::UndefInitializer, dims::Integer...) where {T,N} = CuHostArray{T,N}(undef, dims)
CuHostArray{T}(::UndefInitializer, dims::Dims{N}) where {T,N} = CuHostArray{T,N}(undef, dims)
CuHostArray{T}(::UndefInitializer, dims::Integer...) where {T} =
  CuHostArray{T}(undef, convert(Tuple{Vararg{Int}}, dims))

Base.similar(a::CuHostArray{T,N}) where {T,N} = CuHostArray{T,N}(undef, size(a))


## memory copy operations

function CuHostArray(src::Array{T,N}) where {T,N}
    dst = CuHostArray{T,N}(undef, size(src))
    unsafe_copyto!(pointer(dst), pointer(src), length(src))
    return dst
end

function Base.Array(src::CuHostArray{T,N}) where {T,N}
    dst = Array{T,N}(undef, size(src))
    unsafe_copyto!(pointer(dst), pointer(src), length(src))
    return dst
end


## conversions

using Adapt
function Adapt.adapt_storage(::CUDAnative.Adaptor, a::CuHostArray{T,N}) where {T,N}
    ptr = pointer(a)
    devptr = CUDAnative.DevicePtr{T,AS.Global}(ptr)
    CuDeviceArray{T,N,AS.Global}(a.dims, devptr)
end
