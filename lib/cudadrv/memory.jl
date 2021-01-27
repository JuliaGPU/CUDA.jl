# Raw memory management

export Mem, attribute, attribute!, memory_type, is_managed

module Mem

using ..CUDA
using ..CUDA: @enum_without_prefix, CUstream, CUdevice, CuDim3, CUarray, CUarray_format

using Base: @deprecate_binding

using Printf


#
# buffers
#

# a chunk of memory allocated using the CUDA APIs. this memory can reside on the host, on
# the gpu, or can represent specially-formatted memory (like texture arrays). depending on
# all that, the buffer may be `convert`ed to a Ptr, CuPtr, or CuArrayPtr.

abstract type AbstractBuffer end

Base.convert(T::Type{<:Union{Ptr,CuPtr,CuArrayPtr}}, buf::AbstractBuffer) =
    throw(ArgumentError("Illegal conversion of a $(typeof(buf)) to a $T"))

# ccall integration
#
# taking the pointer of a buffer means returning the underlying pointer,
# and not the pointer of the buffer object itself.
Base.unsafe_convert(T::Type{<:Union{Ptr,CuPtr,CuArrayPtr}}, buf::AbstractBuffer) = convert(T, buf)


## device buffer

"""
    Mem.DeviceBuffer
    Mem.Device

A buffer of device memory residing on the GPU.
"""
struct DeviceBuffer <: AbstractBuffer
    ptr::CuPtr{Cvoid}
    bytesize::Int
end

Base.pointer(buf::DeviceBuffer) = buf.ptr
Base.sizeof(buf::DeviceBuffer) = buf.bytesize

Base.show(io::IO, buf::DeviceBuffer) =
    @printf(io, "DeviceBuffer(%s at %p)", Base.format_bytes(sizeof(buf)), Int(pointer(buf)))

Base.convert(::Type{CuPtr{T}}, buf::DeviceBuffer) where {T} =
    convert(CuPtr{T}, pointer(buf))


"""
    Mem.alloc(DeviceBuffer, bytesize::Integer)

Allocate `bytesize` bytes of memory on the device. This memory is only accessible on the
GPU, and requires explicit calls to `unsafe_copyto!`, which wraps `cuMemcpy`,
for access on the CPU.
"""
function alloc(::Type{DeviceBuffer}, bytesize::Integer;
               async::Bool=false, stream::CuStream=stream())
    bytesize == 0 && return DeviceBuffer(CU_NULL, 0)

    ptr_ref = Ref{CUDA.CUdeviceptr}()
    # FIXME: async allocator never frees memory?
    if false && CUDA.version() >= v"11.2"
        CUDA.cuMemAllocAsync(ptr_ref, bytesize, stream)
        async || synchronize(stream)
    else
        CUDA.cuMemAlloc_v2(ptr_ref, bytesize)
    end

    return DeviceBuffer(reinterpret(CuPtr{Cvoid}, ptr_ref[]), bytesize)
end


function free(buf::DeviceBuffer; async::Bool=false, stream::CuStream=stream())
    pointer(buf) == CU_NULL && return

    # FIXME: async allocator never frees memory?
    if false && CUDA.version() >= v"11.2"
        CUDA.cuMemFreeAsync(buf, stream)
        async || synchronize(stream)
    else
        CUDA.cuMemFree_v2(buf)
    end
end


## host buffer

"""
    Mem.HostBuffer
    Mem.Host

A buffer of pinned memory on the CPU, possibly accessible on the GPU.
"""
struct HostBuffer <: AbstractBuffer
    ptr::Ptr{Cvoid}
    bytesize::Int

    mapped::Bool
end

Base.pointer(buf::HostBuffer) = buf.ptr
Base.sizeof(buf::HostBuffer) = buf.bytesize

Base.show(io::IO, buf::HostBuffer) =
    @printf(io, "HostBuffer(%s at %p)", Base.format_bytes(sizeof(buf)), Int(pointer(buf)))

Base.convert(::Type{Ptr{T}}, buf::HostBuffer) where {T} =
    convert(Ptr{T}, pointer(buf))

function Base.convert(::Type{CuPtr{T}}, buf::HostBuffer) where {T}
    if buf.mapped
        pointer(buf) == C_NULL && return convert(CuPtr{T}, CU_NULL)
        ptr_ref = Ref{CuPtr{Cvoid}}()
        CUDA.cuMemHostGetDevicePointer_v2(ptr_ref, pointer(buf), #=flags=# 0)
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
    bytesize == 0 && return HostBuffer(C_NULL, 0, false)

    ptr_ref = Ref{Ptr{Cvoid}}()
    CUDA.cuMemHostAlloc(ptr_ref, bytesize, flags)

    mapped = (flags & HOSTALLOC_DEVICEMAP) != 0
    return HostBuffer(ptr_ref[], bytesize, mapped)
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

    CUDA.cuMemHostRegister_v2(ptr, bytesize, flags)

    mapped = (flags & HOSTREGISTER_DEVICEMAP) != 0
    return HostBuffer(ptr, bytesize, mapped)
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
struct UnifiedBuffer <: AbstractBuffer
    ptr::CuPtr{Cvoid}
    bytesize::Int
end

Base.pointer(buf::UnifiedBuffer) = buf.ptr
Base.sizeof(buf::UnifiedBuffer) = buf.bytesize

Base.show(io::IO, buf::UnifiedBuffer) =
    @printf(io, "UnifiedBuffer(%s at %p)", Base.format_bytes(sizeof(buf)), Int(pointer(buf)))

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
    bytesize == 0 && return UnifiedBuffer(CU_NULL, 0)

    ptr_ref = Ref{CuPtr{Cvoid}}()
    CUDA.cuMemAllocManaged(ptr_ref, bytesize, flags)

    return UnifiedBuffer(ptr_ref[], bytesize)
end


function free(buf::UnifiedBuffer)
    if pointer(buf) != CU_NULL
        CUDA.cuMemFree_v2(buf)
    end
end


"""
    prefetch(::UnifiedBuffer, [bytes::Integer]; [device::CuDevice], [stream::CuStream])

Prefetches memory to the specified destination device.
"""
function prefetch(buf::UnifiedBuffer, bytes::Integer=sizeof(buf);
                  device::CuDevice=device(), stream::CuStream=stream())
    bytes > sizeof(buf) && throw(BoundsError(buf, bytes))
    CUDA.cuMemPrefetchAsync(buf, bytes, device, stream)
end


@enum_without_prefix CUDA.CUmem_advise CU_MEM_

"""
    advise(::UnifiedBuffer, advice::CUDA.CUmem_advise, [bytes::Integer]; [device::CuDevice])

Advise about the usage of a given memory range.
"""
function advise(buf::UnifiedBuffer, advice::CUDA.CUmem_advise, bytes::Integer=sizeof(buf);
                device::CuDevice=device())
    bytes > sizeof(buf) && throw(BoundsError(buf, bytes))
    CUDA.cuMemAdvise(buf, bytes, advice, device)
end


## array buffer

mutable struct ArrayBuffer{T,N} <: AbstractBuffer
    ptr::CuArrayPtr{T}
    dims::Dims{N}
end

Base.pointer(buf::ArrayBuffer) = buf.ptr
Base.sizeof(buf::ArrayBuffer) = error("Opaque array buffers do not have a definite size")
Base.size(buf::ArrayBuffer) = buf.dims
Base.length(buf::ArrayBuffer) = prod(buf.dims)
Base.ndims(buf::ArrayBuffer{<:Any,N}) where {N} = N

Base.show(io::IO, buf::ArrayBuffer{T,1}) where {T} =
    @printf(io, "%g-element ArrayBuffer{%s,%g}(%p)", length(buf), string(T), 1, Int(pointer(buf)))
Base.show(io::IO, buf::ArrayBuffer{T}) where {T} =
    @printf(io, "%s ArrayBuffer{%s,%g}(%p)", Base.inds2string(size(buf)), string(T), ndims(buf), Int(pointer(buf)))

# array buffers are typed, so refuse arbitrary conversions
Base.convert(::Type{CuArrayPtr{T}}, buf::ArrayBuffer{T}) where {T} =
    convert(CuArrayPtr{T}, pointer(buf))
# ... except for CuArrayPtr{Nothing}, which is used to call untyped API functions
Base.convert(::Type{CuArrayPtr{Nothing}}, buf::ArrayBuffer)  =
    convert(CuArrayPtr{Nothing}, pointer(buf))

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
    CUDA.cuArray3DCreate_v2(handle_ref, allocateArray_ref)
    ptr = reinterpret(CuArrayPtr{T}, handle_ref[])

    return ArrayBuffer{T,N}(ptr, dims)
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
# pointers
#

## initialization

"""
    Mem.set!(buf::CuPtr, value::Union{UInt8,UInt16,UInt32}, len::Integer;
             async::Bool=false, stream::CuStream)

Initialize device memory by copying `val` for `len` times. Executed asynchronously if
`async` is true, otherwise `stream` is synchronized.
"""
set!

for T in [UInt8, UInt16, UInt32]
    bits = 8*sizeof(T)
    fn = Symbol("cuMemsetD$(bits)Async")
    @eval function set!(ptr::CuPtr{$T}, value::$T, len::Integer;
                        async::Bool=false, stream::CuStream=stream())
        $(getproperty(CUDA, fn))(ptr, value, len, stream)
        async || synchronize(stream)
        return
    end
end


## copy operations

for (fn, srcPtrTy, dstPtrTy) in (("cuMemcpyDtoHAsync_v2", CuPtr, Ptr),
                                 ("cuMemcpyHtoDAsync_v2", Ptr,   CuPtr),
                                 ("cuMemcpyDtoDAsync_v2", CuPtr, CuPtr),
                                 )
    @eval function Base.unsafe_copyto!(dst::$dstPtrTy{T}, src::$srcPtrTy{T}, N::Integer;
                                       stream::CuStream=stream(),
                                       async::Bool=false) where T
        $(getproperty(CUDA, Symbol(fn)))(dst, src, N*sizeof(T), stream)
        async || synchronize(stream)
        return dst
    end
end

function Base.unsafe_copyto!(dst::CuArrayPtr{T}, doffs::Integer, src::Ptr{T}, N::Integer;
                             stream::CuStream=stream(),
                             async::Bool=false) where T
    CUDA.cuMemcpyHtoAAsync_v2(dst, doffs, src, N*sizeof(T), stream)
    async || synchronize(stream)
    return dst
end

function Base.unsafe_copyto!(dst::Ptr{T}, src::CuArrayPtr{T}, soffs::Integer, N::Integer;
                             stream::CuStream=stream(),
                             async::Bool=false) where T
    CUDA.cuMemcpyAtoHAsync_v2(dst, src, soffs, N*sizeof(T), stream)
    async || synchronize(stream)
    return dst
end

Base.unsafe_copyto!(dst::CuArrayPtr{T}, doffs::Integer, src::CuPtr{T}, N::Integer) where {T} =
    CUDA.cuMemcpyDtoA_v2(dst, doffs, src, N*sizeof(T))

Base.unsafe_copyto!(dst::CuPtr{T}, src::CuArrayPtr{T}, soffs::Integer, N::Integer) where {T} =
    CUDA.cuMemcpyAtoD_v2(dst, src, soffs, N*sizeof(T))

Base.unsafe_copyto!(dst::CuArrayPtr, src, N::Integer; kwargs...) =
    Base.unsafe_copyto!(dst, 0, src, N; kwargs...)

Base.unsafe_copyto!(dst, src::CuArrayPtr, N::Integer; kwargs...) =
    Base.unsafe_copyto!(dst, src, 0, N; kwargs...)

function unsafe_copy2d!(dst::Union{Ptr{T},CuPtr{T},CuArrayPtr{T}}, dstTyp::Type{<:AbstractBuffer},
                        src::Union{Ptr{T},CuPtr{T},CuArrayPtr{T}}, srcTyp::Type{<:AbstractBuffer},
                        width::Integer, height::Integer=1;
                        dstPos::CuDim=(1,1), srcPos::CuDim=(1,1),
                        dstPitch::Integer=0, srcPitch::Integer=0,
                        async::Bool=false, stream::CuStream=stream()) where T
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
    CUDA.cuMemcpy2DAsync_v2(params_ref, stream)
    async || synchronize(stream)
    return dst
end

"""
    unsafe_copy3d!(dst, dstTyp, src, srcTyp, width, height=1, depth=1;
                   dstPos=(1,1,1), dstPitch=0, dstHeight=0,
                   srcPos=(1,1,1), srcPitch=0, srcHeight=0,
                   async=false, stream=nothing)

Perform a 3D memory copy between pointers `src` and `dst`, at respectively position `srcPos`
and `dstPos` (1-indexed). Both pitch and destination can be specified for both the source
and destination; consult the CUDA documentation for more details. This call is executed
asynchronously if `async` is set, otherwise `stream` is synchronized.
"""
function unsafe_copy3d!(dst::Union{Ptr{T},CuPtr{T},CuArrayPtr{T}}, dstTyp::Type{<:AbstractBuffer},
                        src::Union{Ptr{T},CuPtr{T},CuArrayPtr{T}}, srcTyp::Type{<:AbstractBuffer},
                        width::Integer, height::Integer=1, depth::Integer=1;
                        dstPos::CuDim=(1,1,1), srcPos::CuDim=(1,1,1),
                        dstPitch::Integer=0, dstHeight::Integer=0,
                        srcPitch::Integer=0, srcHeight::Integer=0,
                        async::Bool=false, stream::CuStream=stream()) where T
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
    CUDA.cuMemcpy3DAsync_v2(params_ref, stream)
    async || synchronize(stream)
    return dst
end



#
# auxiliary functionality
#

## memory pinning
# TODO: PerDevice
const __pinned_memory = Dict{Tuple{CuContext,Ptr{Cvoid}}, WeakRef}()
function pin(a::Base.Array, flags=0)
    ctx = context()
    ptr = convert(Ptr{Cvoid}, pointer(a))
    if haskey(__pinned_memory, (ctx,ptr)) && __pinned_memory[(ctx,ptr)].value !== nothing
        return
    end

    buf = Mem.register(Mem.Host, pointer(a), sizeof(a), flags)
    finalizer(a) do _
        CUDA.isvalid(ctx) || return
        context!(ctx) do
            Mem.unregister(buf)
        end
    end
    __pinned_memory[(ctx,ptr)] = WeakRef(a)
    return
end

## memory info

function info()
    free_ref = Ref{Csize_t}()
    total_ref = Ref{Csize_t}()
    CUDA.cuMemGetInfo_v2(free_ref, total_ref)
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
