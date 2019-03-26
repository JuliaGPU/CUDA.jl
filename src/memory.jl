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
    @boundscheck bytes > sizeof(buf) && throw(BoundsError(buf, bytes))
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


## concrete buffer types

# device buffer: residing on the GPU

struct DeviceBuffer <: Buffer
    ptr::CuPtr{Cvoid}
    bytesize::Int
    ctx::CuContext
end

Base.similar(buf::DeviceBuffer, ptr::CuPtr{Cvoid}=pointer(buf),
             bytesize::Int=sizeof(buf), ctx::CuContext=buf.ctx) =
    DeviceBuffer(ptr, bytesize, ctx)

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
    HostBuffer(ptr, bytesize, ctx, buf.flags)

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
    UnifiedBuffer(ptr, bytesize, ctx, buf.flags)

Base.unsafe_convert(::Type{Ptr{T}}, buf::UnifiedBuffer) where {T} =
    convert(Ptr{T}, reinterpret(Ptr{Cvoid}, pointer(buffer)))

Base.unsafe_convert(::Type{CuPtr{T}}, buf::UnifiedBuffer) where {T} =
    convert(CuPtr{T}, pointer(buffer))

# aliases

# for dispatch
const AnyHostBuffer   = Union{HostBuffer,  UnifiedBuffer}
const AnyDeviceBuffer = Union{DeviceBuffer,UnifiedBuffer}

# for convenience
const Device  = DeviceBuffer
const Host    = HostBuffer
const Unified = UnifiedBuffer


## memory management

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


## initialization

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


## copy operations

# TODO: Base.copyto! generally works with number of elements, not number of bytes

# between host memory
Base.unsafe_copyto!(dst::AnyHostBuffer, src::Ref, nbytes::Integer) =
    ccall(:memcpy, Ptr{Cvoid},
          (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
          dst, src, nbytes)

# device to host
Base.unsafe_copyto!(dst::Ref, src::AnyDeviceBuffer, nbytes::Integer) =
    @apicall(:cuMemcpyDtoH,
             (Ptr{Cvoid}, CuPtr{Cvoid}, Csize_t),
             dst, src, nbytes)
Base.unsafe_copyto!(dst::HostBuffer, src::AnyDeviceBuffer, nbytes::Integer,
             stream::CuStream=CuDefaultStream()) =
    @apicall(:cuMemcpyDtoHAsync,
             (Ptr{Cvoid}, CuPtr{Cvoid}, Csize_t, CuStream_t),
             dst, src, nbytes, stream)

# host to device
Base.unsafe_copyto!(dst::AnyDeviceBuffer, src::Ref, nbytes::Integer) =
    @apicall(:cuMemcpyHtoD,
             (CuPtr{Cvoid}, Ptr{Cvoid}, Csize_t),
             dst, src, nbytes)
Base.unsafe_copyto!(dst::AnyDeviceBuffer, src::HostBuffer, nbytes::Integer,
             stream::CuStream=CuDefaultStream()) =
    @apicall(:cuMemcpyHtoDAsync,
             (CuPtr{Cvoid}, Ptr{Cvoid}, Csize_t, CuStream_t),
             dst, src, nbytes, stream)

# between device memory
Base.unsafe_copyto!(dst::AnyDeviceBuffer, src::AnyDeviceBuffer, nbytes::Integer,
             stream::CuStream=CuDefaultStream()) =
    @apicall(:cuMemcpyDtoDAsync,
             (CuPtr{Cvoid}, Ptr{Cvoid}, Csize_t, CuStream_t),
             dst, src, nbytes, stream)


## other

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

function advise(buf::UnifiedBuffer, advice::CUmem_advise, bytes=sizeof(buf),
                device=device(buf.ctx))
    bytes > sizeof(buf) && throw(BoundsError(buf, bytes))
    @apicall(:cuMemAdvise,
             (CuPtr{Cvoid}, Csize_t, Cuint, CuDevice_t),
             buf, bytes, advice, device)
end


## memory info

function info()
    free_ref = Ref{Csize_t}()
    total_ref = Ref{Csize_t}()
    @apicall(:cuMemGetInfo, (Ptr{Csize_t},Ptr{Csize_t}), free_ref, total_ref)
    return convert(Int, free_ref[]), convert(Int, total_ref[])
end

end

"""
    available_memory()

Returns the available_memory amount of memory (in bytes), available for allocation by the CUDA context.
"""
available_memory() = Mem.info()[1]

"""
    total_memory()

Returns the total amount of memory (in bytes), available for allocation by the CUDA context.
"""
total_memory() = Mem.info()[2]
