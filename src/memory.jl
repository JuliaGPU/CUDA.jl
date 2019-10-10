# Raw memory management

export Mem

module Mem

using ..CUDAdrv
import ..@apicall, ..CuStream_t, ..CuDevice_t


## abstract buffer type

abstract type Buffer end

# expected interface:
# - similar()
# - ptr, bytesize and ctx fields
# - convert() to certain pointers

Base.pointer(buf::Buffer) = buf.ptr
Base.sizeof(buf::Buffer) = buf.bytesize

function Base.view(buf::Buffer, bytes::Int)
    @boundscheck bytes > sizeof(buf) && throw(BoundsError(buf, bytes))
    return similar(buf, pointer(buf)+bytes, sizeof(buf)-bytes)
end

# ccall integration
#
# taking the pointer of a buffer means returning the underlying pointer,
# and not the pointer of the buffer object itself.
Base.unsafe_convert(T::Type{<:Union{Ptr,CuPtr}}, buf::Buffer) = convert(T, buf)


## refcounting

const refcounts = Dict{Buffer, Int}()

function refcount(buf::Buffer)
    get(refcounts, buf, 0)
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

Base.convert(::Type{<:Ptr}, buf::DeviceBuffer) =
    throw(ArgumentError("cannot take the CPU address of a GPU buffer"))

Base.convert(::Type{CuPtr{T}}, buf::DeviceBuffer) where {T} =
    convert(CuPtr{T}, pointer(buf))

# host buffer: pinned memory on the CPU, possibly accessible on the GPU

struct HostBuffer <: Buffer
    ptr::Ptr{Cvoid}
    bytesize::Int
    ctx::CuContext

    mapped::Bool
end

Base.similar(buf::HostBuffer, ptr::Ptr{Cvoid}=pointer(buf), bytesize::Int=sizeof(buf),
             ctx::CuContext=buf.ctx, mapped::Bool=buf.mapped) =
    HostBuffer(ptr, bytesize, ctx, mapped)

Base.convert(::Type{Ptr{T}}, buf::HostBuffer) where {T} =
    convert(Ptr{T}, pointer(buf))

function Base.convert(::Type{CuPtr{T}}, buf::HostBuffer) where {T}
    if buf.mapped
        pointer(buf) == C_NULL && return CU_NULL
        ptr_ref = Ref{CuPtr{Cvoid}}()
        @apicall(:cuMemHostGetDevicePointer,
                 (Ptr{CuPtr{Cvoid}}, Ptr{Cvoid}, Cuint),
                 ptr_ref, pointer(buf), #=flags=# 0)
        convert(CuPtr{T}, ptr_ref[])
    else
        throw(ArgumentError("cannot take the GPU address of a pinned but not mapped CPU buffer"))
    end
end

# unified buffer: managed buffer that is accessible on both the CPU and GPU

struct UnifiedBuffer <: Buffer
    ptr::CuPtr{Cvoid}
    bytesize::Int
    ctx::CuContext
end

Base.similar(buf::UnifiedBuffer, ptr::CuPtr{Cvoid}=pointer(buf),
             bytesize::Int=sizeof(buf), ctx::CuContext=buf.ctx) =
    UnifiedBuffer(ptr, bytesize, ctx)

Base.convert(::Type{Ptr{T}}, buf::UnifiedBuffer) where {T} =
    convert(Ptr{T}, reinterpret(Ptr{Cvoid}, pointer(buf)))

Base.convert(::Type{CuPtr{T}}, buf::UnifiedBuffer) where {T} =
    convert(CuPtr{T}, pointer(buf))

# convenience aliases

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
Base.:&(x::CUmem_host_alloc, y::CUmem_host_alloc) =
    reinterpret(CUmem_host_alloc, Base.cconvert(Unsigned, x) & Base.cconvert(Unsigned, y))

"""
    alloc(HostBuffer, bytesize::Integer, [flags])

Allocate `bytesize` bytes of page-locked memory on the host. This memory is accessible from
the CPU, and makes it possible to perform faster memory copies to the GPU. Furthermore, if
`flags` is set to `HOSTALLOC_DEVICEMAP` the memory is also accessible from the GPU. These
accesses are direct, and go through the PCI bus.
"""
function alloc(::Type{HostBuffer}, bytesize::Integer, flags::CUmem_host_alloc=HOSTALLOC_DEFAULT)
    bytesize == 0 && return HostBuffer(C_NULL, 0, CuContext(C_NULL), false)

    ptr_ref = Ref{Ptr{Cvoid}}()
    @apicall(:cuMemHostAlloc,
             (Ptr{Ptr{Cvoid}}, Csize_t, Cuint),
             ptr_ref, bytesize, flags)

    mapped = (flags & HOSTALLOC_DEVICEMAP) != HOSTALLOC_DEFAULT
    return HostBuffer(ptr_ref[], bytesize, CuCurrentContext(), mapped)
end

@enum CUmem_attach::Cuint begin
    ATTACH_GLOBAL   = 0x01  # memory can be accessed by any stream on any device
    ATTACH_HOST     = 0x02  # memory cannot be accessed by any stream on any device
    ATTACH_SINGLE   = 0x04  # memory can only be accessed by a single stream on the associated device
end

# FIXME: EnumSet from JuliaLang/julia#19470
Base.:|(x::CUmem_attach, y::CUmem_attach) =
    reinterpret(CUmem_attach, Base.cconvert(Unsigned, x) | Base.cconvert(Unsigned, y))
Base.:&(x::CUmem_attach, y::CUmem_attach) =
    reinterpret(CUmem_attach, Base.cconvert(Unsigned, x) & Base.cconvert(Unsigned, y))

"""
    alloc(UnifiedBuffer, bytesize::Integer, [flags])

Allocate `bytesize` bytes of unified memory. This memory is accessible from both the CPU and
GPU, with the CUDA driver automatically copying upon first access.
"""
function alloc(::Type{UnifiedBuffer}, bytesize::Integer, flags::CUmem_attach=ATTACH_GLOBAL)
    bytesize == 0 && return UnifiedBuffer(CU_NULL, 0, CuContext(C_NULL))

    ptr_ref = Ref{CuPtr{Cvoid}}()
    @apicall(:cuMemAllocManaged,
             (Ptr{CuPtr{Cvoid}}, Csize_t, Cuint),
             ptr_ref, bytesize, flags)

    return UnifiedBuffer(ptr_ref[], bytesize, CuCurrentContext())
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

@enum CUmem_host_register::Cuint begin
  HOSTREGISTER_DEFAULT   = 0x00
  HOSTREGISTER_PORTABLE  = 0x01 # registered memory will be considered as pinned memory by
                                # all CUDA contexts, not just the one that performed the allocation.
  HOSTREGISTER_DEVICEMAP = 0x02 # maps the allocation into the CUDA address space
  HOSTREGISTER_IOMEMORY  = 0x04 # pointer is treated as pointing to some I/O memory space,
                                # e.g. the PCI Express resource of a 3rd party device.
end

# FIXME: EnumSet from JuliaLang/julia#19470
Base.:|(x::CUmem_host_register, y::CUmem_host_register) =
    reinterpret(CUmem_host_register, Base.cconvert(Unsigned, x) | Base.cconvert(Unsigned, y))
Base.:&(x::CUmem_host_register, y::CUmem_host_register) =
    reinterpret(CUmem_host_register, Base.cconvert(Unsigned, x) & Base.cconvert(Unsigned, y))

"""
    register(HostBuffer, ptr::Ptr, bytesize::Integer, [flags::CUmem_host_register])

Page-lock the host memory pointed to by `ptr`. Subsequent transfers to and from devices will
be faster, and can be executed asynchronously. If the `HOSTREGISTER_DEVICEMAP` flag is
specified, the buffer will also be accessible directly from the GPU. These accesses are
direct, and go through the PCI bus.
"""
function register(::Type{HostBuffer}, ptr::Ptr, bytesize::Integer,
                  flags::CUmem_host_register=HOSTREGISTER_DEFAULT)
    bytesize == 0 && throw(ArgumentError())

    @apicall(:cuMemHostRegister,
             (Ptr{Cvoid}, Csize_t, Cuint),
             ptr, bytesize, flags)

    mapped = (flags & HOSTREGISTER_DEVICEMAP) != HOSTREGISTER_DEFAULT
    return HostBuffer(ptr, bytesize, CuCurrentContext(), mapped)
end

function unregister(buf::HostBuffer)
    @apicall(:cuMemHostUnregister, (Ptr{Cvoid},), buf)
end


## initialization

const AnyHostBuffer = Union{Ref,HostBuffer,UnifiedBuffer}
const AnyDeviceBuffer = Union{DeviceBuffer,UnifiedBuffer}

"""
    set!(buf::DeviceBuffer, value::Union{UInt8,UInt16,UInt32}, len::Integer;
         async::Bool=false, stream::CuStream)

Initialize device memory by copying `val` for `len` times. Executed asynchronously if
`async` is true, in which case a valid `stream` is required.
"""
set!

for T in [UInt8, UInt16, UInt32]
    bits = 8*sizeof(T)
    fn_sync = Symbol("cuMemsetD$(bits)")
    fn_async = Symbol("cuMemsetD$(bits)Async")
    @eval function set!(buf::AnyDeviceBuffer, value::$T, len::Integer;
                        async::Bool=false, stream::Union{Nothing,CuStream}=nothing)
        if async
          stream===nothing &&
              throw(ArgumentError("Asynchronous memory operations require a stream."))
            @apicall($(QuoteNode(fn_async)),
                     (CuPtr{Cvoid}, $T, Csize_t, CuStream_t),
                     buf, value, len, stream)
        else
          stream===nothing ||
              throw(ArgumentError("Synchronous memory operations cannot be issues on a stream."))
            @apicall($(QuoteNode(fn_sync)),
                     (CuPtr{Cvoid}, $T, Csize_t),
                     buf, value, len)
        end
    end
end


## copy operations

# NOTE: we don't extend `Base.unsafe_copyto!` because that function works with elements

# NOTE: we don't have a memcpy implementation for host to host buffers for multiple reasons:
#       1) this package is a CUDA API wrapper, the user can just convert to a CPU pointer;
#       2) you sometimes want to do an explicit CUD API memcpy even for unified memory,
#          since otherwise you need to synchronize the device first.
#          https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-gpu-exclusive
for (f, dstTy, srcTy) in (("cuMemcpyDtoH", Union{Ref,HostBuffer}, AnyDeviceBuffer),
                          ("cuMemcpyHtoD", AnyDeviceBuffer, Union{Ref,HostBuffer}),
                          ("cuMemcpyDtoD", AnyDeviceBuffer, AnyDeviceBuffer),
                         )
    dstPtrTy = (dstTy==Union{Ref,HostBuffer} ? Ptr : CuPtr)
    srcPtrTy = (srcTy==Union{Ref,HostBuffer} ? Ptr : CuPtr)

    @eval function copy!(dst::$dstTy, src::$srcTy, nbytes::Integer;
                         stream::Union{Nothing,CuStream}=nothing,
                         async::Bool=false)
        if async
            stream===nothing &&
                throw(ArgumentError("Asynchronous memory operations require a stream."))
            @apicall($(QuoteNode(Symbol(f * "Async"))),
                     ($dstPtrTy{Cvoid}, $srcPtrTy{Cvoid}, Csize_t, CuStream_t),
                     dst, src, nbytes, stream)
        else
            stream===nothing ||
                throw(ArgumentError("Synchronous memory operations cannot be issued on a stream."))
            @apicall($(QuoteNode(Symbol(f))),
                     ($dstPtrTy{Cvoid}, $srcPtrTy{Cvoid}, Csize_t),
                     dst, src, nbytes)
        end
        return dst
    end
end

"""
  copy!(dst, src, nbytes::Integer)
  copy!(dst, src, nbytes::Integer; async::Bool=false, stream::CuStream)

Copy `nbytes` bytes of memory from `src` to `dst`. The source and destination arguments can
point to memory managed by CUDA, represented by objects of type `Buffer`, or any other piece
of memory using `Ref` objects.

When copying to or from a GPU device, additional keyword arguments are supported: `async` to
control whether the operation should be asynchronous, and `stream` to indicate on which
stream the asynchronous operation should happen (this argument is not supported when `async`
is set to false). Note that memory operations will only truly behave asynchronously when
other conditions are met, e.g., if any CPU memory is allocated as a page-locked buffer.
"""
copy!


## unified memory

function prefetch(buf::UnifiedBuffer, bytes=sizeof(buf);
                  device::CuDevice=device(), stream::CuStream=CuDefaultStream())
    bytes > sizeof(buf) && throw(BoundsError(buf, bytes))
    @apicall(:cuMemPrefetchAsync,
             (CuPtr{Cvoid}, Csize_t, CuDevice_t, CuStream_t),
             buf, bytes, device, stream)
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
