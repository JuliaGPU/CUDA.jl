# Raw memory management

# TODO:
# - cuMemHostRegister to page-lock existing buffers
# - consistent CPU/GPU or host/device terminology

export Mem

module Mem

using ..CUDAdrv
import ..@apicall, ..CuStream_t, ..CuDevice_t


## abstract buffer type

abstract type Buffer <: Ref{Cvoid} end

# expected interface:
# - similar()
# - ptr, bytesize and ctx fields
# - unsafe_convert() to certain pointers

Base.pointer(buf::Buffer) = buf.ptr
Base.sizeof(buf::Buffer) = buf.bytesize

function Base.view(buf::Buffer, bytes::Int)
    bytes > sizeof(buf) && throw(BoundsError(buf, bytes))
    return similar(buf, pointer(buf)+bytes, sizeof(buf)-bytes)
end



## refcounting

const refcounts = Dict{Buffer, Int}()

function refcount(buf::Buffer)
    get(refcounts, Base.unsafe_convert(CuPtr{Cvoid}, buf), 0)
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


## concrete buffer types

# device buffer: residing on the GPU

struct DeviceBuffer <: Buffer
    ptr::CuPtr{Cvoid}
    bytesize::Int
    ctx::CuContext
end

Base.similar(buf::DeviceBuffer, ptr::CuPtr{Cvoid}=pointer(buf),
             bytesize::Int=sizeof(buf), ctx::CuContext=buf.ctx) =
    DeviceBuffer(bytesize, ptr, ctx)

Base.unsafe_convert(::Type{<:Ptr}, buf::DeviceBuffer) =
    throw(ArgumentError("cannot take the CPU address of a GPU buffer"))

Base.unsafe_convert(::Type{CuPtr{T}}, buf::DeviceBuffer) where {T} =
    convert(CuPtr{T}, pointer(buf))

# host buffer: pinned memory on the CPU, possibly accessible on the GPU

@enum CUmem_host_alloc::Cuint begin
    HOSTALLOC_DEFAULT       = 0x00
    HOSTALLOC_PORTABLE      = 0x01  # memory is portable between CUDA contexts
    HOSTALLOC_DEVICEMAP     = 0x02  # memory is mapped into CUDA address space and
                                    # cuMemHostGetDevicePointer may be called on the pointer
    HOSTALLOC_WRITECOMBINED = 0x04  # memory is allocated as write-combined - fast to write,
                                    # faster to DMA, slow to read except via SSE4 MOVNTDQA
end

# FIXME: EnumSet from JuliaLang/julia#19470
Base.:|(x::CUmem_host_alloc, y::CUmem_host_alloc) =
    reinterpret(CUmem_host_alloc, Base.cconvert(Unsigned, x) | Base.cconvert(Unsigned, y))

struct HostBuffer <: Buffer
    ptr::Ptr{Cvoid}
    bytesize::Int
    ctx::CuContext

    flags::CUmem_host_alloc
end

Base.similar(buf::HostBuffer, ptr::Ptr{Cvoid}=pointer(buf),
             bytesize::Int=sizeof(buf), ctx::CuContext=buf.ctx,
             flags::CUmem_host_alloc=buf.flags) =
    HostBuffer(bytesize, ptr, ctx, buf.flags)

Base.unsafe_convert(::Type{Ptr{T}}, buf::HostBuffer) where {T} =
    convert(Ptr{T}, pointer(buffer))

function Base.unsafe_convert(::Type{CuPtr{T}}, buf::HostBuffer) where {T}
    if buf.flags & HOSTALLOC_DEVICEMAP
        pointer(buf) == C_NULL && return CU_NULL
        ptr_ref[] = Ref{CuPtr{Cvoid}}()
        @apicall(:cuMemHostGetDevicePointer,
                 (Ptr{CuPtr{Cvoid}}, Ptr{Cvoid}, Cuint),
                 ptr_ref, pointer(buf), #=flags=# 0)
        convert(CuPtr{T}, ptr_ref[])
    else
        throw(ArgumentError("cannot take the GPU address of a pinned but not mapped CPU buffer"))
    end
end

# unified buffer: managed buffer that is accessible on both the CPU and GPU

@enum CUmem_attach::Cuint begin
    ATTACH_GLOBAL   = 0x01  # memory can be accessed by any stream on any device
    ATTACH_HOST     = 0x02  # memory cannot be accessed by any stream on any device
    ATTACH_SINGLE   = 0x04  # memory can only be accessed by a single stream on the associated device
end

# FIXME: EnumSet from JuliaLang/julia#19470
Base.:|(x::CUmem_attach, y::CUmem_attach) =
    reinterpret(CUmem_attach, Base.cconvert(Unsigned, x) | Base.cconvert(Unsigned, y))

struct UnifiedBuffer <: Buffer
    ptr::CuPtr{Cvoid}
    bytesize::Int
    ctx::CuContext

    flags::CUmem_attach
end

Base.similar(buf::UnifiedBuffer, ptr::CuPtr{Cvoid}=pointer(buf),
             bytesize::Int=sizeof(buf), ctx::CuContext=buf.ctx,
             flags::CUmem_attach=buf.flags) =
    UnifiedBuffer(bytesize, ptr, ctx, buf.flags)

Base.unsafe_convert(::Type{Ptr{T}}, buf::UnifiedBuffer) where {T} =
    convert(Ptr{T}, reinterpret(Ptr{Cvoid}, pointer(buffer)))

Base.unsafe_convert(::Type{CuPtr{T}}, buf::UnifiedBuffer) where {T} =
    convert(CuPtr{T}, pointer(buffer))

# aliases for dispatch

const AnyHostBuffer   = Union{HostBuffer,  UnifiedBuffer}
const AnyDeviceBuffer = Union{DeviceBuffer,UnifiedBuffer}


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
    alloc(DeviceBuffer, bytesize::Integer)

Allocate `bytesize` bytes of memory on the device. This memory is only accessible on the
GPU, and requires explicit calls to `upload` and `download` for access on the CPU.
"""
function alloc(::Type{DeviceBuffer}, bytesize::Integer)
    bytesize == 0 && return DeviceBuffer(CU_NULL, 0, CuContext(C_NULL))

    ptr_ref = Ref{CuPtr{Cvoid}}()
    @apicall(:cuMemAlloc,
             (Ptr{CuPtr{Cvoid}}, Csize_t),
             ptr_ref, bytesize)

    return DeviceBuffer(ptr_ref[], bytesize, CuCurrentContext())
end

"""
    alloc(HostBuffer, bytesize::Integer, [flags])

Allocate `bytesize` bytes of page-locked memory on the host. This memory is accessible from
the CPU, and makes it possible to perform faster memory copies to the GPU. Furthermore, if
`flags` is set to `HOSTALLOC_DEVICEMAP` the memory is also accessible from the GPU. These
accesses are direct, and go through the PCI bus.
"""
function alloc(::Type{HostBuffer}, bytesize::Integer, flags::CUmem_host_alloc=HOSTALLOC_DEFAULT)
    bytesize == 0 && return HostBuffer(C_NULL, 0, CuContext(C_NULL), flags)

    ptr_ref = Ref{Ptr{Cvoid}}()
    @apicall(:cuMemHostAlloc,
             (Ptr{Ptr{Cvoid}}, Csize_t, Cuint),
             ptr_ref, bytesize, flags)

    return HostBuffer(ptr_ref[], bytesize, CuCurrentContext(), flags)
end

"""
    alloc(UnifiedBuffer, bytesize::Integer, [flags])

Allocate `bytesize` bytes of unified memory. This memory is accessible from both the CPU and
GPU, with the CUDA driver automatically copying upon first access.
"""
function alloc(::Type{UnifiedBuffer}, bytesize::Integer, flags::CUmem_attach=ATTACH_GLOBAL)
    bytesize == 0 && return UnifiedBuffer(C_NULL, 0, CuContext(C_NULL), flags)

    ptr_ref = Ref{Ptr{Cvoid}}()
    @apicall(:cuMemAllocManaged,
             (Ptr{Ptr{Cvoid}}, Csize_t, Cuint),
             ptr_ref, bytesize, flags)

    return UnifiedBuffer(ptr_ref[], bytesize, CuCurrentContext(), flags)
end

function prefetch(buf::DeviceBuffer, bytes=sizeof(buf); stream::CuStream=CuDefaultStream())
    bytes > sizeof(buf) && throw(BoundsError(buf, bytes))
    dev = device(buf.ctx)
    @apicall(:cuMemPrefetchAsync,
             (CuPtr{Cvoid}, Csize_t, CuDevice_t, CuStream_t),
             buf, bytes, dev, stream)
end

@enum CUmem_advise::Cuint begin
    ADVISE_SET_READ_MOSTLY          = 0x01  # data will mostly be read and only occasionally be written to
    ADVISE_UNSET_READ_MOSTLY        = 0x02  #
    ADVISE_SET_PREFERRED_LOCATION   = 0x03  # set the preferred location for the data as the specified device
    ADVISE_UNSET_PREFERRED_LOCATION = 0x04  #
    ADVISE_SET_ACCESSED_BY          = 0x05  # data will be accessed by the specified device,
                                            # so prevent page faults as much as possible
    ADVISE_UNSET_ACCESSED_BY        = 0x06  #
end

function advise(buf::UnifiedBuffer, advice::CUmem_advise, bytes=sizeof(buf), device=device(buf.ctx))
    bytes > sizeof(buf) && throw(BoundsError(buf, bytes))
    @apicall(:cuMemAdvise,
             (CuPtr{Cvoid}, Csize_t, Cuint, CuDevice_t),
             buf, bytes, advice, device)
end

function free(buf::Union{DeviceBuffer,UnifiedBuffer})
    if pointer(buf) != CU_NULL
        @apicall(:cuMemFree, (CuPtr{Cvoid},), buf)
    end
end

function free(buf::HostBuffer)
    if pointer(buf) != CU_NULL
        @apicall(:cuMemFreeHost, (Ptr{Cvoid},), buf)
    end
end

for T in [UInt8, UInt16, UInt32]
    bits = 8*sizeof(T)
    fn_sync = Symbol("cuMemsetD$(bits)")
    fn_async = Symbol("cuMemsetD$(bits)Async")
    @eval begin
        @doc $"""
            set!(buf::DeviceBuffer, value::$T, len::Integer, [stream=CuDefaultStream()]; async=false)

        Initialize device memory by copying the $bits-bit value `val` for `len` times.
        Executed asynchronously if `async` is true.
        """
        function set!(buf::DeviceBuffer, value::$T, len::Integer,
                      stream::CuStream=CuDefaultStream(); async::Bool=false)
            if async
                @apicall($(QuoteNode(fn_async)),
                         (CuPtr{Cvoid}, $T, Csize_t, CuStream_t),
                         buf, value, len, stream)
            else
                @assert stream==CuDefaultStream()
                @apicall($(QuoteNode(fn_sync)),
                         (CuPtr{Cvoid}, $T, Csize_t),
                         buf, value, len)
            end
        end
    end
end

"""
    upload!(dst::Buffer,  src, nbytes::Integer, [stream=CuDefaultStream()]; async=false)

Upload `nbytes` memory from `src` at the host to `dst` on the device.
"""
function upload!(dst::AnyDeviceBuffer, src::Ref, nbytes::Integer,
                 stream::CuStream=CuDefaultStream(); async::Bool=false)
    if async
        isa(src, HostBuffer) ||
            @warn "Cannot asynchronously upload from non-pinned host memory" maxlog=1
        @apicall(:cuMemcpyHtoDAsync,
                 (CuPtr{Cvoid}, Ptr{Cvoid}, Csize_t, CuStream_t),
                 dst, src, nbytes, stream)
    else
        stream==CuDefaultStream() ||
            throw(ArgumentError("Cannot specify a stream for synchronous operations."))
        @apicall(:cuMemcpyHtoD,
                 (CuPtr{Cvoid}, Ptr{Cvoid}, Csize_t),
                 dst, src, nbytes)
    end
end

"""
    download!(dst::Ref, src::DeviceBuffer,  nbytes::Integer, [stream=CuDefaultStream()]; async=false)
    download!(dst::Ref, src::UnifiedBuffer, nbytes::Integer, [stream=CuDefaultStream()]; async=false)

Download `nbytes` memory from `src` on the device to `src` on the host.
"""
function download!(dst::Ref, src::AnyDeviceBuffer, nbytes::Integer,
                   stream::CuStream=CuDefaultStream(); async::Bool=false)
    if async
        isa(dst, HostBuffer) ||
            @warn "Cannot asynchronously upload to non-pinned host memory" maxlog=1
        @apicall(:cuMemcpyDtoHAsync,
                 (Ptr{Cvoid}, CuPtr{Cvoid}, Csize_t, CuStream_t),
                 dst, src, nbytes, stream)
    else
        stream==CuDefaultStream() ||
            throw(ArgumentError("Cannot specify a stream for synchronous operations."))
        @apicall(:cuMemcpyDtoH,
                 (Ptr{Cvoid}, CuPtr{Cvoid}, Csize_t),
                 dst, src, nbytes)
    end
end

"""
    transfer!(dst::Buffer, src::Buffer, nbytes::Integer, [stream=CuDefaultStream()]; async=false)

Transfer `nbytes` of device memory from `src` to `dst`.
"""
function transfer!(dst::AnyDeviceBuffer, src::AnyDeviceBuffer, nbytes::Integer,
                   stream::CuStream=CuDefaultStream(); async::Bool=false)
    if async
        @apicall(:cuMemcpyDtoDAsync,
                 (CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t, CuStream_t),
                 dst, src, nbytes, stream)
    else
        stream==CuDefaultStream() ||
            throw(ArgumentError("Cannot specify a stream for synchronous operations."))
        @apicall(:cuMemcpyDtoD,
                 (CuPtr{Cvoid}, CuPtr{Cvoid}, Csize_t),
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
function upload(src::AbstractArray) # TODO: stream, async
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
    if isa(T, UnionAll) || T.abstract || !isconcretetype(T)
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
    dst = Vector{T}(undef, count)
    download!(dst, src, stream; async=async)
    return dst
end


## deprecations

function alloc(bytesize::Integer, managed=false; flags::CUmem_attach=ATTACH_GLOBAL)
    if managed
        Base.depwarn("`Mem.alloc(bytesize, managed=true; [flags...])` is deprecated, use `Mem.alloc(UnifiedBuffer, bytesize, [flags...])` instead.", :alloc)
        alloc(UnifiedByffer, bytesize, flags)
    else
        Base.depwarn("`Mem.alloc(bytesize, [managed=false])` is deprecated, use `Mem.alloc(DeviceBuffer, bytesize)` instead.", :alloc)
        alloc(DeviceBuffer, bytesize)
    end
end

end
