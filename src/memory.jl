# Raw memory management

export Mem, attribute, attribute!, memory_type, is_managed

module Mem

using ..CUDAdrv
using ..CUDAdrv: @enum_without_prefix, CUstream, CUdevice

using Base: @deprecate_binding


#
# untyped buffers
#

abstract type Buffer end

# expected interface:
# - similar()
# - ptr, bytesize and ctx fields
# - convert() to certain pointers

CUDAdrv.device(buf::Buffer) = device(buf.ctx)

Base.sizeof(buf::Buffer) = buf.bytesize

# ccall integration
#
# taking the pointer of a buffer means returning the underlying pointer,
# and not the pointer of the buffer object itself.
Base.unsafe_convert(T::Type{<:Union{Ptr,CuPtr}}, buf::Buffer) = convert(T, buf)


## device buffer

"""
    Mem.DeviceBuffer
    Mem.Device

A buffer of device memory residing on the GPU.
"""
struct DeviceBuffer <: Buffer
    ptr::CuPtr{Cvoid}
    bytesize::Int
    ctx::CuContext
end

Base.similar(buf::DeviceBuffer, ptr::CuPtr{Cvoid}=buf.ptr,
             bytesize::Int=sizeof(buf), ctx::CuContext=buf.ctx) =
    DeviceBuffer(ptr, bytesize, ctx)

Base.convert(::Type{<:Ptr}, buf::DeviceBuffer) =
    throw(ArgumentError("cannot take the CPU address of a GPU buffer"))

Base.convert(::Type{CuPtr{T}}, buf::DeviceBuffer) where {T} =
    convert(CuPtr{T}, buf.ptr)


"""
    Mem.alloc(DeviceBuffer, bytesize::Integer)

Allocate `bytesize` bytes of memory on the device. This memory is only accessible on the
GPU, and requires explicit calls to `unsafe_copyto!`, which wraps `cuMemcpy`,
for access on the CPU.
"""
function alloc(::Type{DeviceBuffer}, bytesize::Integer)
    bytesize == 0 && return DeviceBuffer(CU_NULL, 0, CuContext(C_NULL))

    ptr_ref = Ref{CUDAdrv.CUdeviceptr}()
    CUDAdrv.cuMemAlloc(ptr_ref, bytesize)

    return DeviceBuffer(reinterpret(CuPtr{Cvoid}, ptr_ref[]), bytesize, CuCurrentContext())
end


function free(buf::DeviceBuffer)
    if buf.ptr != CU_NULL
        CUDAdrv.cuMemFree(buf)
    end
end


## host buffer

"""
    Mem.HostBuffer
    Mem.Host

A buffer of pinned memory on the CPU, possible accessible on the GPU.
"""
struct HostBuffer <: Buffer
    ptr::Ptr{Cvoid}
    bytesize::Int
    ctx::CuContext

    mapped::Bool
end

Base.similar(buf::HostBuffer, ptr::Ptr{Cvoid}=buf.ptr, bytesize::Int=sizeof(buf),
             ctx::CuContext=buf.ctx, mapped::Bool=buf.mapped) =
    HostBuffer(ptr, bytesize, ctx, mapped)

Base.convert(::Type{Ptr{T}}, buf::HostBuffer) where {T} =
    convert(Ptr{T}, buf.ptr)

function Base.convert(::Type{CuPtr{T}}, buf::HostBuffer) where {T}
    if buf.mapped
        buf.ptr == C_NULL && return convert(CuPtr{T}, CU_NULL)
        ptr_ref = Ref{CuPtr{Cvoid}}()
        CUDAdrv.cuMemHostGetDevicePointer(ptr_ref, buf.ptr, #=flags=# 0)
        convert(CuPtr{T}, ptr_ref[])
    else
        throw(ArgumentError("cannot take the GPU address of a pinned but not mapped CPU buffer"))
    end
end


@deprecate_binding HOSTALLOC_DEFAULT 0 false
const HOSTALLOC_PORTABLE = CUDAdrv.CU_MEMHOSTALLOC_PORTABLE
const HOSTALLOC_DEVICEMAP = CUDAdrv.CU_MEMHOSTALLOC_DEVICEMAP
const HOSTALLOC_WRITECOMBINED = CUDAdrv.CU_MEMHOSTALLOC_WRITECOMBINED

"""
    Mem.alloc(HostBuffer, bytesize::Integer, [flags])

Allocate `bytesize` bytes of page-locked memory on the host. This memory is accessible from
the CPU, and makes it possible to perform faster memory copies to the GPU. Furthermore, if
`flags` is set to `HOSTALLOC_DEVICEMAP` the memory is also accessible from the GPU.
These accesses are direct, and go through the PCI bus.
If `flags` is set to `HOSTALLOC_PORTABLE`, the memory is considered mapped by all CUDA contexts,
not just the one that created the memory, which is useful if the memory needs to be accessed from
multiple devices. Multiple `flags` can be set at one time using a bytewise `OR`:

    flags = HOSTALLOC_PORTABLE | HOSTALLOC_DEVICEMAP

"""
function alloc(::Type{HostBuffer}, bytesize::Integer, flags=0)
    bytesize == 0 && return HostBuffer(C_NULL, 0, CuContext(C_NULL), false)

    ptr_ref = Ref{Ptr{Cvoid}}()
    CUDAdrv.cuMemHostAlloc(ptr_ref, bytesize, flags)

    mapped = (flags & HOSTALLOC_DEVICEMAP) != 0
    return HostBuffer(ptr_ref[], bytesize, CuCurrentContext(), mapped)
end


const HOSTREGISTER_PORTABLE = CUDAdrv.CU_MEMHOSTREGISTER_PORTABLE
const HOSTREGISTER_DEVICEMAP = CUDAdrv.CU_MEMHOSTREGISTER_DEVICEMAP
const HOSTREGISTER_IOMEMORY = CUDAdrv.CU_MEMHOSTREGISTER_IOMEMORY

"""
    Mem.register(HostBuffer, ptr::Ptr, bytesize::Integer, [flags])

Page-lock the host memory pointed to by `ptr`. Subsequent transfers to and from devices will
be faster, and can be executed asynchronously. If the `HOSTREGISTER_DEVICEMAP` flag is
specified, the buffer will also be accessible directly from the GPU.
These accesses are direct, and go through the PCI bus.
If the `HOSTREGISTER_PORTABLE` flag is specified, any CUDA context can access the memory.
"""
function register(::Type{HostBuffer}, ptr::Ptr, bytesize::Integer, flags=0)
    bytesize == 0 && throw(ArgumentError("Cannot register an empty range of memory."))

    CUDAdrv.cuMemHostRegister(ptr, bytesize, flags)

    mapped = (flags & HOSTREGISTER_DEVICEMAP) != 0
    return HostBuffer(ptr, bytesize, CuCurrentContext(), mapped)
end

"""
    Mem.unregister(HostBuffer)

Unregisters a memory range that was registered with [`Mem.register`](@ref).
"""
function unregister(buf::HostBuffer)
    CUDAdrv.cuMemHostUnregister(buf)
end


function free(buf::HostBuffer)
    if buf.ptr != CU_NULL
        CUDAdrv.cuMemFreeHost(buf)
    end
end


## unified buffer

"""
    Mem.UnifiedBuffer
    Mem.Unified

A managed buffer that is accessible on both the CPU and GPU.
"""
struct UnifiedBuffer <: Buffer
    ptr::CuPtr{Cvoid}
    bytesize::Int
    ctx::CuContext
end

Base.similar(buf::UnifiedBuffer, ptr::CuPtr{Cvoid}=buf.ptr,
             bytesize::Int=sizeof(buf), ctx::CuContext=buf.ctx) =
    UnifiedBuffer(ptr, bytesize, ctx)

Base.convert(::Type{Ptr{T}}, buf::UnifiedBuffer) where {T} =
    convert(Ptr{T}, reinterpret(Ptr{Cvoid}, buf.ptr))

Base.convert(::Type{CuPtr{T}}, buf::UnifiedBuffer) where {T} =
    convert(CuPtr{T}, buf.ptr)

@enum_without_prefix CUDAdrv.CUmemAttach_flags CU_MEM_

"""
    Mem.alloc(UnifiedBuffer, bytesize::Integer, [flags::CUmemAttach_flags])

Allocate `bytesize` bytes of unified memory. This memory is accessible from both the CPU and
GPU, with the CUDA driver automatically copying upon first access.
"""
function alloc(::Type{UnifiedBuffer}, bytesize::Integer,
              flags::CUDAdrv.CUmemAttach_flags=ATTACH_GLOBAL)
    bytesize == 0 && return UnifiedBuffer(CU_NULL, 0, CuContext(C_NULL))

    ptr_ref = Ref{CuPtr{Cvoid}}()
    CUDAdrv.cuMemAllocManaged(ptr_ref, bytesize, flags)

    return UnifiedBuffer(ptr_ref[], bytesize, CuCurrentContext())
end


function free(buf::UnifiedBuffer)
    if buf.ptr != CU_NULL
        CUDAdrv.cuMemFree(buf)
    end
end


"""
    prefetch(::UnifiedBuffer, [bytes::Integer]; [device::CuDevice], [stream::CuStream])

Prefetches memory to the specified destination device.
"""
function prefetch(buf::UnifiedBuffer, bytes::Integer=sizeof(buf);
                  device::CuDevice=device(buf), stream::CuStream=CuDefaultStream())
    bytes > sizeof(buf) && throw(BoundsError(buf, bytes))
    CUDAdrv.cuMemPrefetchAsync(buf, bytes, device, stream)
end


@enum_without_prefix CUDAdrv.CUmem_advise CU_MEM_

"""
    advise(::UnifiedBuffer, advice::CUDAdrv.CUmem_advise, [bytes::Integer]; [device::CuDevice])

Advise about the usage of a given memory range.
"""
function advise(buf::UnifiedBuffer, advice::CUDAdrv.CUmem_advise, bytes::Integer=sizeof(buf);
                device::CuDevice=device(buf))
    bytes > sizeof(buf) && throw(BoundsError(buf, bytes))
    CUDAdrv.cuMemAdvise(buf, bytes, advice, device)
end


## convenience aliases

const Device  = DeviceBuffer
const Host    = HostBuffer
const Unified = UnifiedBuffer



#
# typed pointers
#

## initialization

"""
    Mem.set!(buf::CuPtr, value::Union{UInt8,UInt16,UInt32}, len::Integer;
             async::Bool=false, stream::CuStream)

Initialize device memory by copying `val` for `len` times. Executed asynchronously if
`async` is true, in which case a valid `stream` is required.
"""
set!

for T in [UInt8, UInt16, UInt32]
    bits = 8*sizeof(T)
    fn_sync = Symbol("cuMemsetD$(bits)")
    fn_async = Symbol("cuMemsetD$(bits)Async")
    @eval function set!(ptr::CuPtr{$T}, value::$T, len::Integer;
                        async::Bool=false, stream::Union{Nothing,CuStream}=nothing)
        if async
          stream===nothing &&
              throw(ArgumentError("Asynchronous memory operations require a stream."))
            $(getproperty(CUDAdrv, fn_async))(ptr, value, len, stream)
        else
          stream===nothing ||
              throw(ArgumentError("Synchronous memory operations cannot be issues on a stream."))
            $(getproperty(CUDAdrv, fn_sync))(ptr, value, len)
        end
    end
end


## copy operations

for (f, srcPtrTy, dstPtrTy) in (("cuMemcpyDtoH", CuPtr, Ptr),
                                ("cuMemcpyHtoD", Ptr,   CuPtr),
                                ("cuMemcpyDtoD", CuPtr, CuPtr),
                               )
    @eval function Base.unsafe_copyto!(dst::$dstPtrTy{T}, src::$srcPtrTy{T}, N::Integer;
                                       stream::Union{Nothing,CuStream}=nothing,
                                       async::Bool=false) where T
        if async
            stream===nothing &&
                throw(ArgumentError("Asynchronous memory operations require a stream."))
            $(getproperty(CUDAdrv, Symbol(f * "Async")))(dst, src, N*sizeof(T), stream)
        else
            stream===nothing ||
                throw(ArgumentError("Synchronous memory operations cannot be issued on a stream."))
            $(getproperty(CUDAdrv, Symbol(f)))(dst, src, N*sizeof(T))
        end
        return dst
    end
end


## memory info

function info()
    free_ref = Ref{Csize_t}()
    total_ref = Ref{Csize_t}()
    CUDAdrv.cuMemGetInfo(free_ref, total_ref)
    return convert(Int, free_ref[]), convert(Int, total_ref[])
end

end # module Mem

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


## pointer attributes

"""
    attribute(X, ptr::Union{Ptr,CuPtr}, attr)

Returns attribute `attr` about pointer `ptr`. The type of the returned value depends on the
attribute, and as such must be passed as the `X` parameter.
"""
function attribute(X::Type, ptr::Union{Ptr{T},CuPtr{T}}, attr::CUpointer_attribute) where {T}
    ptr = reinterpret(CuPtr{T}, ptr)
    data_ref = Ref{X}()
    cuPointerGetAttribute(data_ref, attr, ptr)
    return data_ref[]
end

"""
    attribute!(ptr::Union{Ptr,CuPtr}, attr, val)

Sets attribute` attr` on a pointer `ptr` to `val`.
"""
function attribute!(ptr::Union{Ptr{T},CuPtr{T}}, attr::CUpointer_attribute, val) where {T}
    ptr = reinterpret(CuPtr{T}, ptr)
    cuPointerSetAttribute(Ref(val), attr, ptr)
    return
end

@enum_without_prefix CUpointer_attribute CU_

# some common attributes

@enum_without_prefix CUmemorytype CU_
memory_type(x) = CUmemorytype(attribute(Cuint, x, POINTER_ATTRIBUTE_MEMORY_TYPE))

is_managed(x) = convert(Bool, attribute(Cuint, x, POINTER_ATTRIBUTE_IS_MANAGED))
