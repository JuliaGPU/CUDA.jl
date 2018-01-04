# Raw memory management

export Mem

module Mem

using CUDAdrv
import CUDAdrv: @apicall, CuStream_t

using Compat


## buffer type

struct Buffer
    ptr::Ptr{Cvoid}
    bytesize::Int

    ctx::CuContext
end

Base.unsafe_convert(::Type{Ptr{T}}, buf::Buffer) where {T} = convert(Ptr{T}, buf.ptr)

Base.isnull(buf::Buffer) = (buf.ptr == C_NULL)

function view(buf::Buffer, bytes::Int)
    bytes > buf.bytesize && throw(BoundsError(buf, bytes))
    return Mem.Buffer(buf.ptr+bytes, buf.bytesize-bytes, buf.ctx)
end



## refcounting

const refcounts = Dict{Buffer, Int}()

function refcount(buf::Buffer)
    get(refcounts, Base.unsafe_convert(Ptr{Cvoid}, buf), 0)
end

"""
    retain(buf)

Increase the refcount of a buffer.
"""
function retain(buf::Buffer)
    refcount = get!(refcounts, buf, 0)
    refcounts[buf] = refcount + 1
    return
end

"""
    release(buf)

Decrease the refcount of a buffer. Returns `true` if the refcount has dropped to 0, and
some action needs to be taken.
"""
function release(buf::Buffer)
    haskey(refcounts, buf) || error("Release of unmanaged $buf")
    refcount = refcounts[buf]
    @assert refcount > 0 "Release of dead $buf"
    refcounts[buf] = refcount - 1
    return refcount==1
end


## memory info

"""
    info()

Returns a tuple of two integers, indicating respectively the free and total amount of memory
(in bytes) available for allocation by the CUDA context.
"""
function info()
    free_ref = Ref{Csize_t}()
    total_ref = Ref{Csize_t}()
    @apicall(:cuMemGetInfo, (Ptr{Csize_t},Ptr{Csize_t}), free_ref, total_ref)
    return convert(Int, free_ref[]), convert(Int, total_ref[])
end

"""
    free()

Returns the free amount of memory (in bytes), available for allocation by the CUDA context.
"""
free() = info()[1]

"""
    total()

Returns the total amount of memory (in bytes), available for allocation by the CUDA context.
"""
total() = info()[2]

"""
    used()

Returns the used amount of memory (in bytes), allocated by the CUDA context.
"""
used() = total()-free()


## generic interface (for documentation purposes)

"""
Allocate linear memory on the device and return a buffer to the allocated memory. The
allocated memory is suitably aligned for any kind of variable. The memory will not be freed
automatically, use [`free(::Buffer)`](@ref) for that.
"""
function alloc end

"""
Free device memory.
"""
function free end

"""
Initialize device memory with a repeating value.
"""
function set! end

"""
Upload memory from host to device.
Executed asynchronously on `stream` if `async` is true.
"""
function upload end
@doc (@doc upload) upload!

"""
Download memory from device to host.
Executed asynchronously on `stream` if `async` is true.
"""
function download end
@doc (@doc download) download!

"""
Transfer memory from device to device.
Executed asynchronously on `stream` if `async` is true.
"""
function transfer end
@doc (@doc transfer) transfer!


## pointer-based

"""
    alloc(bytes::Integer)

Allocate `bytesize` bytes of memory.
"""
function alloc(bytesize::Integer)
    bytesize == 0 && throw(ArgumentError("invalid amount of memory requested"))

    ptr_ref = Ref{Ptr{Cvoid}}()
    @apicall(:cuMemAlloc, (Ptr{Ptr{Cvoid}}, Csize_t), ptr_ref, bytesize)
    return Buffer(ptr_ref[], bytesize, CuCurrentContext())
end

function free(buf::Buffer)
    @apicall(:cuMemFree, (Ptr{Cvoid},), buf.ptr)
    return
end

for T in [UInt8, UInt16, UInt32]
    bits = 8*sizeof(T)
    fn_sync = Symbol("cuMemsetD$(bits)")
    fn_async = Symbol("cuMemsetD$(bits)Async")
    @eval begin
        @doc $"""
            set!(buf::Buffer, value::$T, len::Integer, [stream=CuDefaultStream()]; async=false)

        Initialize device memory by copying the $bits-bit value `val` for `len` times.
        Executed asynchronously if `async` is true.
        """
        function set!(buf::Buffer, value::$T, len::Integer,
                      stream::CuStream=CuDefaultStream(); async::Bool=false)
            if async
                @apicall($(QuoteNode(fn_async)),
                         (Ptr{Cvoid}, $T, Csize_t, CuStream_t),
                         buf.ptr, value, len, stream)
            else
                @assert stream==CuDefaultStream()
                @apicall($(QuoteNode(fn_sync)),
                         (Ptr{Cvoid}, $T, Csize_t),
                         buf.ptr, value, len)
            end
        end
    end
end

"""
    upload!(dst::Buffer, src, nbytes::Integer, [stream=CuDefaultStream()]; async=false)

Upload `nbytes` memory from `src` at the host to `dst` on the device.
"""
function upload!(dst::Buffer, src::Ref, nbytes::Integer,
                 stream::CuStream=CuDefaultStream(); async::Bool=false)
    if async
        @apicall(:cuMemcpyHtoDAsync,
                 (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, CuStream_t),
                 dst, src, nbytes, stream)
    else
        @assert stream==CuDefaultStream()
        @apicall(:cuMemcpyHtoD,
                 (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
                 dst, src, nbytes)
    end
end

"""
    download!(dst::Ref, src::Buffer, nbytes::Integer, [stream=CuDefaultStream()]; async=false)

Download `nbytes` memory from `src` on the device to `src` on the host.
"""
function download!(dst::Ref, src::Buffer, nbytes::Integer,
                   stream::CuStream=CuDefaultStream(); async::Bool=false)
    if async
        @apicall(:cuMemcpyDtoHAsync,
                 (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, CuStream_t),
                 dst, src, nbytes, stream)
    else
        @assert stream==CuDefaultStream()
        @apicall(:cuMemcpyDtoH,
                 (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
                 dst, src, nbytes)
    end
end

"""
    transfer!(dst::Buffer, src::Buffer, nbytes::Integer, [stream=CuDefaultStream()]; async=false)

Transfer `nbytes` of device memory from `src` to `dst`.
"""
function transfer!(dst::Buffer, src::Buffer, nbytes::Integer,
                   stream::CuStream=CuDefaultStream(); async::Bool=false)
    if async
        @apicall(:cuMemcpyDtoDAsync,
                 (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, CuStream_t),
                 dst, src, nbytes, stream)
    else
        @assert stream==CuDefaultStream()
        @apicall(:cuMemcpyDtoD,
                 (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
                 dst, src, nbytes)
    end
end


## array based

"""
    alloc(src::AbstractArray)

Allocate space to store the contents of `src`.
"""
function alloc(src::AbstractArray)
    return alloc(sizeof(src))
end

"""
    upload!(dst::Buffer, src::AbstractArray, [stream=CuDefaultStream()]; async=false)

Upload the contents of an array `src` to `dst`.
"""
function upload!(dst::Buffer, src::AbstractArray,
                 stream=CuDefaultStream(); async::Bool=false)
    upload!(dst, Ref(src, 1), sizeof(src), stream; async=async)
end

"""
    upload(src::AbstractArray)::Buffer

Allocates space for and uploads the contents of an array `src`, returning a Buffer.
Cannot be executed asynchronously due to the synchronous allocation.
"""
function upload(src::AbstractArray)
    dst = alloc(src)
    upload!(dst, src)
    return dst
end

"""
    download!(dst::AbstractArray, src::Buffer, [stream=CuDefaultStream()]; async=false)

Downloads memory from `src` to the array at `dst`. The amount of memory downloaded is
determined by calling `sizeof` on the array, so it needs to be properly preallocated.
"""
function download!(dst::AbstractArray, src::Buffer,
                   stream::CuStream=CuDefaultStream(); async::Bool=false)
    ref = Ref(dst, 1)
    download!(ref, src, sizeof(dst), stream; async=async)
    return
end


## type based

function check_type(::Type{Buffer}, T)
    if isa(T, UnionAll) || T.abstract || !T.isleaftype
        throw(ArgumentError("cannot represent abstract or non-leaf object"))
    end
    Base.datatype_pointerfree(T) || throw(ArgumentError("cannot handle non-ptrfree objects"))
    sizeof(T) == 0 && throw(ArgumentError("cannot represent singleton objects"))
end

"""
    alloc(T::Type, [count::Integer=1])

Allocate space for `count` objects of type `T`.
"""
function alloc(::Type{T}, count::Integer=1) where {T}
    check_type(Buffer, T)

    return alloc(sizeof(T)*count)
end

"""
    download(::Type{T}, src::Buffer, [count::Integer=1], [stream=CuDefaultStream()]; async=false)::Vector{T}

Download `count` objects of type `T` from the device at `src`, returning a vector.
"""
function download(::Type{T}, src::Buffer, count::Integer=1,
                  stream::CuStream=CuDefaultStream(); async::Bool=false) where {T}
    dst = Vector{T}(uninitialized, count)
    download!(dst, src, stream; async=async)
    return dst
end

end
