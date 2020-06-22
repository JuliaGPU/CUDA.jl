# Raw memory management

export Mem, attribute, attribute!, memory_type, is_managed

module Mem

using ..CUDA
using ..CUDA: @enum_without_prefix, CUstream, CUdevice, CuDim3, CUarray, CUarray_format

using Base: @deprecate_binding


# TODO: needs another redesign
#
# - Buffers should be typed, since ArrayBuffers are tied to a format.
#   untyped buffers can use T=UInt8 or T=Nothing and reinterpret the output pointer
#   (this is impossible with textures, though)
# - copyto! methods should take a Buffer so that we can populate the memcpy structs directly
# - allocate buffers on construction, instead of using an alloc method? free on finalizer?
# - have CuArray contain a buffer, use that for dispatch.
# - trait to determine which pointers a buffer can yield?


#
# untyped buffers
#

abstract type Buffer end

# expected interface:
# - similar()
# - ptr, bytesize and ctx fields
# - convert() to Ptr and CuPtr

Base.pointer(buf::Buffer) = buf.ptr

Base.sizeof(buf::Buffer) = buf.bytesize

CUDA.device(buf::Buffer) = device(buf.ctx)

# ccall integration
#
# taking the pointer of a buffer means returning the underlying pointer,
# and not the pointer of the buffer object itself.
Base.unsafe_convert(T::Type{<:Union{Ptr,CuPtr,CuArrayPtr}}, buf::Buffer) = convert(T, buf)


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

Base.similar(buf::DeviceBuffer, ptr::CuPtr{Cvoid}=pointer(buf),
             bytesize::Int=sizeof(buf), ctx::CuContext=buf.ctx) =
    DeviceBuffer(ptr, bytesize, ctx)

Base.convert(::Type{<:Ptr}, buf::DeviceBuffer) =
    throw(ArgumentError("cannot take the CPU address of a GPU buffer"))

Base.convert(::Type{CuPtr{T}}, buf::DeviceBuffer) where {T} =
    convert(CuPtr{T}, pointer(buf))


"""
    Mem.alloc(DeviceBuffer, bytesize::Integer)

Allocate `bytesize` bytes of memory on the device. This memory is only accessible on the
GPU, and requires explicit calls to `unsafe_copyto!`, which wraps `cuMemcpy`,
for access on the CPU.
"""
function alloc(::Type{DeviceBuffer}, bytesize::Integer)
    bytesize == 0 && return DeviceBuffer(CU_NULL, 0, CuContext(C_NULL))

    ptr_ref = Ref{CUDA.CUdeviceptr}()
    CUDA.cuMemAlloc(ptr_ref, bytesize)

    return DeviceBuffer(reinterpret(CuPtr{Cvoid}, ptr_ref[]), bytesize, CuCurrentContext())
end


function free(buf::DeviceBuffer)
    if pointer(buf) != CU_NULL
        CUDA.cuMemFree(buf)
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

Base.similar(buf::HostBuffer, ptr::Ptr{Cvoid}=pointer(buf), bytesize::Int=sizeof(buf),
             ctx::CuContext=buf.ctx, mapped::Bool=buf.mapped) =
    HostBuffer(ptr, bytesize, ctx, mapped)

Base.convert(::Type{Ptr{T}}, buf::HostBuffer) where {T} =
    convert(Ptr{T}, pointer(buf))

function Base.convert(::Type{CuPtr{T}}, buf::HostBuffer) where {T}
    if buf.mapped
        pointer(buf) == C_NULL && return convert(CuPtr{T}, CU_NULL)
        ptr_ref = Ref{CuPtr{Cvoid}}()
        CUDA.cuMemHostGetDevicePointer(ptr_ref, pointer(buf), #=flags=# 0)
        convert(CuPtr{T}, ptr_ref[])
    else
        throw(ArgumentError("cannot take the GPU address of a pinned but not mapped CPU buffer"))
    end
end


@deprecate_binding HOSTALLOC_DEFAULT 0 false
const HOSTALLOC_PORTABLE = CUDA.CU_MEMHOSTALLOC_PORTABLE
const HOSTALLOC_DEVICEMAP = CUDA.CU_MEMHOSTALLOC_DEVICEMAP
const HOSTALLOC_WRITECOMBINED = CUDA.CU_MEMHOSTALLOC_WRITECOMBINED

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
    CUDA.cuMemHostAlloc(ptr_ref, bytesize, flags)

    mapped = (flags & HOSTALLOC_DEVICEMAP) != 0
    return HostBuffer(ptr_ref[], bytesize, CuCurrentContext(), mapped)
end


const HOSTREGISTER_PORTABLE = CUDA.CU_MEMHOSTREGISTER_PORTABLE
const HOSTREGISTER_DEVICEMAP = CUDA.CU_MEMHOSTREGISTER_DEVICEMAP
const HOSTREGISTER_IOMEMORY = CUDA.CU_MEMHOSTREGISTER_IOMEMORY

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

    CUDA.cuMemHostRegister(ptr, bytesize, flags)

    mapped = (flags & HOSTREGISTER_DEVICEMAP) != 0
    return HostBuffer(ptr, bytesize, CuCurrentContext(), mapped)
end

"""
    Mem.unregister(HostBuffer)

Unregisters a memory range that was registered with [`Mem.register`](@ref).
"""
function unregister(buf::HostBuffer)
    CUDA.cuMemHostUnregister(buf)
end


function free(buf::HostBuffer)
    if pointer(buf) != CU_NULL
        CUDA.cuMemFreeHost(buf)
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

Base.similar(buf::UnifiedBuffer, ptr::CuPtr{Cvoid}=pointer(buf),
             bytesize::Int=sizeof(buf), ctx::CuContext=buf.ctx) =
    UnifiedBuffer(ptr, bytesize, ctx)

Base.convert(::Type{Ptr{T}}, buf::UnifiedBuffer) where {T} =
    convert(Ptr{T}, reinterpret(Ptr{Cvoid}, pointer(buf)))

Base.convert(::Type{CuPtr{T}}, buf::UnifiedBuffer) where {T} =
    convert(CuPtr{T}, pointer(buf))

@enum_without_prefix CUDA.CUmemAttach_flags CU_MEM_

"""
    Mem.alloc(UnifiedBuffer, bytesize::Integer, [flags::CUmemAttach_flags])

Allocate `bytesize` bytes of unified memory. This memory is accessible from both the CPU and
GPU, with the CUDA driver automatically copying upon first access.
"""
function alloc(::Type{UnifiedBuffer}, bytesize::Integer,
              flags::CUDA.CUmemAttach_flags=ATTACH_GLOBAL)
    bytesize == 0 && return UnifiedBuffer(CU_NULL, 0, CuContext(C_NULL))

    ptr_ref = Ref{CuPtr{Cvoid}}()
    CUDA.cuMemAllocManaged(ptr_ref, bytesize, flags)

    return UnifiedBuffer(ptr_ref[], bytesize, CuCurrentContext())
end


function free(buf::UnifiedBuffer)
    if pointer(buf) != CU_NULL
        CUDA.cuMemFree(buf)
    end
end


"""
    prefetch(::UnifiedBuffer, [bytes::Integer]; [device::CuDevice], [stream::CuStream])

Prefetches memory to the specified destination device.
"""
function prefetch(buf::UnifiedBuffer, bytes::Integer=sizeof(buf);
                  device::CuDevice=device(buf), stream::CuStream=CuDefaultStream())
    bytes > sizeof(buf) && throw(BoundsError(buf, bytes))
    CUDA.cuMemPrefetchAsync(buf, bytes, device, stream)
end


@enum_without_prefix CUDA.CUmem_advise CU_MEM_

"""
    advise(::UnifiedBuffer, advice::CUDA.CUmem_advise, [bytes::Integer]; [device::CuDevice])

Advise about the usage of a given memory range.
"""
function advise(buf::UnifiedBuffer, advice::CUDA.CUmem_advise, bytes::Integer=sizeof(buf);
                device::CuDevice=device(buf))
    bytes > sizeof(buf) && throw(BoundsError(buf, bytes))
    CUDA.cuMemAdvise(buf, bytes, advice, device)
end


## array buffer

mutable struct ArrayBuffer{T,N} <: Buffer
    ptr::CuArrayPtr{T}
    dims::Dims{N}
    ctx::CuContext
end

Base.convert(::Type{CuArrayPtr{T}}, buf::ArrayBuffer) where {T} =
    convert(CuArrayPtr{T}, pointer(buf))

function alloc(::Type{<:ArrayBuffer{T}}, dims::Dims{N}) where {T,N}
    format = convert(CUarray_format, eltype(T))

    if N == 2
        width, height = dims
        depth = 0
        @assert 1 <= width "CUDA 2D array (texture) width must be >= 1"
        # @assert witdh <= CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH
        @assert 1 <= height "CUDA 2D array (texture) height must be >= 1"
        # @assert height <= CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT
    elseif N == 3
        width, height, depth = dims
        @assert 1 <= width "CUDA 3D array (texture) width must be >= 1"
        # @assert witdh <= CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH
        @assert 1 <= height "CUDA 3D array (texture) height must be >= 1"
        # @assert height <= CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT
        @assert 1 <= depth "CUDA 3D array (texture) depth must be >= 1"
        # @assert depth <= CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH
    elseif N == 1
        width = dims[1]
        height = depth = 0
        @assert 1 <= width "CUDA 1D array (texture) width must be >= 1"
        # @assert witdh <= CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH
    else
        "CUDA arrays (texture memory) can only have 1, 2 or 3 dimensions"
    end

    allocateArray_ref = Ref(CUDA.CUDA_ARRAY3D_DESCRIPTOR(
        width, # Width::Csize_t
        height, # Height::Csize_t
        depth, # Depth::Csize_t
        format, # Format::CUarray_format
        UInt32(CUDA.nchans(T)), # NumChannels::UInt32
        0))

    handle_ref = Ref{CUarray}()
    CUDA.cuArray3DCreate(handle_ref, allocateArray_ref)
    ptr = reinterpret(CuArrayPtr{T}, handle_ref[])

    return ArrayBuffer{T,N}(ptr, dims, CuCurrentContext())
end

function free(buf::ArrayBuffer)
    CUDA.cuArrayDestroy(buf)
end


## convenience aliases

const Device  = DeviceBuffer
const Host    = HostBuffer
const Unified = UnifiedBuffer
const Array   = ArrayBuffer



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
            $(getproperty(CUDA, fn_async))(ptr, value, len, stream)
        else
          stream===nothing ||
              throw(ArgumentError("Synchronous memory operations cannot be issues on a stream."))
            $(getproperty(CUDA, fn_sync))(ptr, value, len)
        end
    end
end


## copy operations

for (f, srcPtrTy, dstPtrTy) in (("cuMemcpyDtoH", CuPtr,      Ptr),
                                ("cuMemcpyHtoD", Ptr,        CuPtr),
                                ("cuMemcpyDtoD", CuPtr,      CuPtr),
                               )
    @eval function Base.unsafe_copyto!(dst::$dstPtrTy{T}, src::$srcPtrTy{T}, N::Integer;
                                       stream::Union{Nothing,CuStream}=nothing,
                                       async::Bool=false) where T
        if async
            stream===nothing &&
                throw(ArgumentError("Asynchronous memory operations require a stream."))
            $(getproperty(CUDA, Symbol(f * "Async")))(dst, src, N*sizeof(T), stream)
        else
            stream===nothing ||
                throw(ArgumentError("Synchronous memory operations cannot be issued on a stream."))
            $(getproperty(CUDA, Symbol(f)))(dst, src, N*sizeof(T))
        end
        return dst
    end
end

function Base.unsafe_copyto!(dst::CuArrayPtr{T}, doffs::Integer, src::Ptr{T}, N::Integer;
                             stream::Union{Nothing,CuStream}=nothing,
                             async::Bool=false) where T
    if async
        stream===nothing &&
            throw(ArgumentError("Asynchronous memory operations require a stream."))
        CUDA.cuMemcpyHtoAAsync(dst, doffs, src, N*sizeof(T), stream)
    else
        stream===nothing ||
            throw(ArgumentError("Synchronous memory operations cannot be issued on a stream."))
        CUDA.cuMemcpyHtoA(dst, doffs, src, N*sizeof(T))
    end
end

function Base.unsafe_copyto!(dst::Ptr{T}, src::CuArrayPtr{T}, soffs::Integer, N::Integer;
                             stream::Union{Nothing,CuStream}=nothing,
                             async::Bool=false) where T
    if async
        stream===nothing &&
            throw(ArgumentError("Asynchronous memory operations require a stream."))
        CUDA.cuMemcpyAtoHAsync(dst, src, soffs, N*sizeof(T), stream)
    else
        stream===nothing ||
            throw(ArgumentError("Synchronous memory operations cannot be issued on a stream."))
        CUDA.cuMemcpyAtoH(dst, src, soffs, N*sizeof(T))
    end
end

Base.unsafe_copyto!(dst::CuArrayPtr{T}, doffs::Integer, src::CuPtr{T}, N::Integer) where {T} =
    CUDA.cuMemcpyDtoA(dst, doffs, src, N*sizeof(T))

Base.unsafe_copyto!(dst::CuPtr{T}, src::CuArrayPtr{T}, soffs::Integer, N::Integer) where {T} =
    CUDA.cuMemcpyAtoD(dst, src, soffs, N*sizeof(T))

Base.unsafe_copyto!(dst::CuArrayPtr, src, N::Integer; kwargs...) =
    Base.unsafe_copyto!(dst, 0, src, N; kwargs...)

Base.unsafe_copyto!(dst, src::CuArrayPtr, N::Integer; kwargs...) =
    Base.unsafe_copyto!(dst, src, 0, N; kwargs...)

function unsafe_copy2d!(dst::Union{Ptr{T},CuPtr{T},CuArrayPtr{T}}, dstTyp::Type{<:Buffer},
                        src::Union{Ptr{T},CuPtr{T},CuArrayPtr{T}}, srcTyp::Type{<:Buffer},
                        width::Integer, height::Integer=1;
                        dstPos::CuDim=(1,1), srcPos::CuDim=(1,1),
                        dstPitch::Integer=0, srcPitch::Integer=0,
                        async::Bool=false, stream::Union{Nothing,CuStream}=nothing) where T
    srcPos = CUDA.CuDim3(srcPos)
    @assert srcPos.z == 1
    dstPos = CUDA.CuDim3(dstPos)
    @assert dstPos.z == 1

    srcMemoryType, srcHost, srcDevice, srcArray = if srcTyp == Host
        CUDA.CU_MEMORYTYPE_HOST,
        src::Ptr,
        0,
        0
    elseif srcTyp == Mem.Device
        CUDA.CU_MEMORYTYPE_DEVICE,
        0,
        src::CuPtr,
        0
    elseif srcTyp == Mem.Unified
        CUDA.CU_MEMORYTYPE_UNIFIED,
        0,
        reinterpret(CuPtr{Cvoid}, src),
        0
    elseif srcTyp == Mem.Array
        CUDA.CU_MEMORYTYPE_ARRAY,
        0,
        0,
        src::CuArrayPtr
    end

    dstMemoryType, dstHost, dstDevice, dstArray = if dstTyp == Host
        CUDA.CU_MEMORYTYPE_HOST,
        dst::Ptr,
        0,
        0
    elseif dstTyp == Mem.Device
        CUDA.CU_MEMORYTYPE_DEVICE,
        0,
        dst::CuPtr,
        0
    elseif dstTyp == Mem.Unified
        CUDA.CU_MEMORYTYPE_UNIFIED,
        0,
        reinterpret(CuPtr{Cvoid}, dst),
        0
    elseif dstTyp == Mem.Array
        CUDA.CU_MEMORYTYPE_ARRAY,
        0,
        0,
        dst::CuArrayPtr
    end

    params_ref = Ref(CUDA.CUDA_MEMCPY2D(
        # source
        (srcPos.x-1)*sizeof(T), srcPos.y-1,
        srcMemoryType, srcHost, srcDevice, srcArray,
        srcPitch,
        # destination
        (dstPos.x-1)*sizeof(T), dstPos.y-1,
        dstMemoryType, dstHost, dstDevice, dstArray,
        dstPitch,
        # extent
        width*sizeof(T), height
    ))
    if async
        stream===nothing &&
            throw(ArgumentError("Asynchronous memory operations require a stream."))
        CUDA.cuMemcpy2DAsync(params_ref, stream)
    else
        stream===nothing ||
            throw(ArgumentError("Synchronous memory operations cannot be issued on a stream."))
        CUDA.cuMemcpy2D(params_ref)
    end
end

"""
    unsafe_copy3d!(dst, dstTyp, src, srcTyp, width, height=1, depth=1;
                   dstPos=(1,1,1), dstPitch=0, dstHeight=0,
                   srcPos=(1,1,1), srcPitch=0, srcHeight=0,
                   async=false, stream=nothing)

Perform a 3D memory copy between pointers `src` and `dst`, at respectively position `srcPos`
and `dstPos` (1-indexed). Both pitch and destination can be specified for both the source
and destination; consult the CUDA documentation for more details. This call is executed
asynchronously if `async` is set, in which case `stream` needs to be a valid CuStream.
"""
function unsafe_copy3d!(dst::Union{Ptr{T},CuPtr{T},CuArrayPtr{T}}, dstTyp::Type{<:Buffer},
                        src::Union{Ptr{T},CuPtr{T},CuArrayPtr{T}}, srcTyp::Type{<:Buffer},
                        width::Integer, height::Integer=1, depth::Integer=1;
                        dstPos::CuDim=(1,1,1), srcPos::CuDim=(1,1,1),
                        dstPitch::Integer=0, dstHeight::Integer=0,
                        srcPitch::Integer=0, srcHeight::Integer=0,
                        async::Bool=false, stream::Union{Nothing,CuStream}=nothing) where T
    srcPos = CUDA.CuDim3(srcPos)
    dstPos = CUDA.CuDim3(dstPos)

    srcMemoryType, srcHost, srcDevice, srcArray = if srcTyp == Host
        CUDA.CU_MEMORYTYPE_HOST,
        src::Ptr,
        0,
        0
    elseif srcTyp == Mem.Device
        CUDA.CU_MEMORYTYPE_DEVICE,
        0,
        src::CuPtr,
        0
    elseif srcTyp == Mem.Unified
        CUDA.CU_MEMORYTYPE_UNIFIED,
        0,
        reinterpret(CuPtr{Cvoid}, src),
        0
    elseif srcTyp == Mem.Array
        CUDA.CU_MEMORYTYPE_ARRAY,
        0,
        0,
        src::CuArrayPtr
    end

    dstMemoryType, dstHost, dstDevice, dstArray = if dstTyp == Host
        CUDA.CU_MEMORYTYPE_HOST,
        dst::Ptr,
        0,
        0
    elseif dstTyp == Mem.Device
        CUDA.CU_MEMORYTYPE_DEVICE,
        0,
        dst::CuPtr,
        0
    elseif dstTyp == Mem.Unified
        CUDA.CU_MEMORYTYPE_UNIFIED,
        0,
        reinterpret(CuPtr{Cvoid}, dst),
        0
    elseif dstTyp == Mem.Array
        CUDA.CU_MEMORYTYPE_ARRAY,
        0,
        0,
        dst::CuArrayPtr
    end

    params_ref = Ref(CUDA.CUDA_MEMCPY3D(
        # source
        (srcPos.x-1)*sizeof(T), srcPos.y-1, srcPos.z-1,
        0, # LOD
        srcMemoryType, srcHost, srcDevice, srcArray,
        C_NULL, # reserved
        srcPitch, srcHeight,
        # destination
        (dstPos.x-1)*sizeof(T), dstPos.y-1, dstPos.z-1,
        0, # LOD
        dstMemoryType, dstHost, dstDevice, dstArray,
        C_NULL, # reserved
        dstPitch, dstHeight,
        # extent
        width*sizeof(T), height, depth
    ))
    if async
        stream===nothing &&
            throw(ArgumentError("Asynchronous memory operations require a stream."))
        CUDA.cuMemcpy3DAsync(params_ref, stream)
    else
        stream===nothing ||
            throw(ArgumentError("Synchronous memory operations cannot be issued on a stream."))
        CUDA.cuMemcpy3D(params_ref)
    end
end



#
# auxiliary functionality
#

## memory pinning
const __pinned_memory = Dict{Ptr, WeakRef}()
function pin(a::Base.Array, flags=0)
    # use pointer instead of objectid?
    ptr = pointer(a)
    if haskey(__pinned_memory, ptr) && __pinned_memory[ptr].value !== nothing
        return nothing
    end
    ad = Mem.register(Mem.Host, pointer(a), sizeof(a), flags)
    finalizer(_ -> Mem.unregister(ad), a)
    __pinned_memory[ptr] = WeakRef(a)
    return nothing
end

## memory info

function info()
    free_ref = Ref{Csize_t}()
    total_ref = Ref{Csize_t}()
    CUDA.cuMemGetInfo(free_ref, total_ref)
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


## shared texture/array stuff

function Base.convert(::Type{CUarray_format}, T::Type)
    if T === UInt8
        return CU_AD_FORMAT_UNSIGNED_INT8
    elseif T === UInt16
        return CU_AD_FORMAT_UNSIGNED_INT16
    elseif T === UInt32
        return CU_AD_FORMAT_UNSIGNED_INT32
    elseif T === Int8
        return CU_AD_FORMAT_SIGNED_INT8
    elseif T === Int16
        return CU_AD_FORMAT_SIGNED_INT16
    elseif T === Int32
        return CU_AD_FORMAT_SIGNED_INT32
    elseif T === Float16
        return CU_AD_FORMAT_HALF
    elseif T === Float32
        return CU_AD_FORMAT_FLOAT
    else
        throw(ArgumentError("CUDA does not support texture arrays for element type $T."))
    end
end

nchans(::Type{<:NTuple{C}}) where {C} = C
nchans(::Type) = 1
