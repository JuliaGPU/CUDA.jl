# Raw memory management

export attribute, attribute!, memory_type, is_managed


#
# operations on memory
#

# a chunk of memory allocated using the CUDA APIs. this memory can reside on the host, on
# the GPU, or can represent specially-formatted memory (like texture arrays). depending on
# all that, the memory object may be `convert`ed to a Ptr, CuPtr, or CuArrayPtr.

abstract type AbstractMemory end

Base.convert(T::Type{<:Union{Ptr,CuPtr,CuArrayPtr}}, mem::AbstractMemory) =
    throw(ArgumentError("Illegal conversion of a $(typeof(mem)) to a $T"))

# ccall integration
#
# taking the pointer of a buffer means returning the underlying pointer,
# and not the pointer of the buffer object itself.
Base.unsafe_convert(T::Type{<:Union{Ptr,CuPtr,CuArrayPtr}}, mem::AbstractMemory) = convert(T, mem)


## device memory

"""
    DeviceMemory

Device memory residing on the GPU.
"""
struct DeviceMemory <: AbstractMemory
    dev::CuDevice
    ctx::CuContext
    ptr::CuPtr{Cvoid}
    bytesize::Int

    async::Bool
end

DeviceMemory() = DeviceMemory(device(), context(), CU_NULL, 0, false)

Base.pointer(mem::DeviceMemory) = mem.ptr
Base.sizeof(mem::DeviceMemory) = mem.bytesize

Base.show(io::IO, mem::DeviceMemory) =
    @printf(io, "DeviceMemory(%s at %p)", Base.format_bytes(sizeof(mem)), Int(pointer(mem)))

Base.convert(::Type{CuPtr{T}}, mem::DeviceMemory) where {T} =
    convert(CuPtr{T}, pointer(mem))

"""
    alloc(DeviceMemory, bytesize::Integer;
          [async=false], [stream::CuStream], [pool::CuMemoryPool])

Allocate `bytesize` bytes of memory on the device. This memory is only accessible on the
GPU, and requires explicit calls to `unsafe_copyto!`, which wraps `cuMemcpy`,
for access on the CPU.
"""
function alloc(::Type{DeviceMemory}, bytesize::Integer;
               async::Bool=memory_pools_supported(device()),
               stream::Union{Nothing,CuStream}=nothing,
               pool::Union{Nothing,CuMemoryPool}=nothing)
    bytesize == 0 && return DeviceMemory()

    ptr_ref = Ref{CUdeviceptr}()
    if async
        stream = @something stream CUDA.stream()
        if pool !== nothing
            cuMemAllocFromPoolAsync(ptr_ref, bytesize, pool, stream)
        else
            cuMemAllocAsync(ptr_ref, bytesize, stream)
        end
    else
        cuMemAlloc_v2(ptr_ref, bytesize)
    end

    return DeviceMemory(device(), context(), reinterpret(CuPtr{Cvoid}, ptr_ref[]), bytesize, async)
end

function free(mem::DeviceMemory; stream::Union{Nothing,CuStream}=nothing)
    pointer(mem) == CU_NULL && return

    if mem.async
        stream = @something stream CUDA.stream()
        cuMemFreeAsync(mem, stream)
    else
        cuMemFree_v2(mem)
    end
end


## host memory

"""
    HostMemory

Pinned memory residing on the CPU, possibly accessible on the GPU.
"""
struct HostMemory <: AbstractMemory
    ctx::CuContext
    ptr::Ptr{Cvoid}
    bytesize::Int
end

HostMemory() = HostMemory(context(), C_NULL, 0)

Base.pointer(mem::HostMemory) = mem.ptr
Base.sizeof(mem::HostMemory) = mem.bytesize

Base.show(io::IO, mem::HostMemory) =
    @printf(io, "HostMemory(%s at %p)", Base.format_bytes(sizeof(mem)), Int(pointer(mem)))

Base.convert(::Type{Ptr{T}}, mem::HostMemory) where {T} =
    convert(Ptr{T}, pointer(mem))

function Base.convert(::Type{CuPtr{T}}, mem::HostMemory) where {T}
    pointer(mem) == C_NULL && return convert(CuPtr{T}, CU_NULL)
    ptr_ref = Ref{CuPtr{Cvoid}}()
    cuMemHostGetDevicePointer_v2(ptr_ref, pointer(mem), #=flags=# 0)
    convert(CuPtr{T}, ptr_ref[])
end


const MEMHOSTALLOC_PORTABLE = CU_MEMHOSTALLOC_PORTABLE
const MEMHOSTALLOC_DEVICEMAP = CU_MEMHOSTALLOC_DEVICEMAP
const MEMHOSTALLOC_WRITECOMBINED = CU_MEMHOSTALLOC_WRITECOMBINED

"""
    alloc(HostMemory, bytesize::Integer, [flags])

Allocate `bytesize` bytes of page-locked memory on the host. This memory is accessible from
the CPU, and makes it possible to perform faster memory copies to the GPU. Furthermore, if
`flags` is set to `MEMHOSTALLOC_DEVICEMAP` the memory is also accessible from the GPU. These
accesses are direct, and go through the PCI bus. If `flags` is set to
`MEMHOSTALLOC_PORTABLE`, the memory is considered mapped by all CUDA contexts, not just the
one that created the memory, which is useful if the memory needs to be accessed from
multiple devices. Multiple `flags` can be set at one time using a bytewise `OR`:

    flags = MEMHOSTALLOC_PORTABLE | MEMHOSTALLOC_DEVICEMAP

"""
function alloc(::Type{HostMemory}, bytesize::Integer, flags=0)
    bytesize == 0 && return HostMemory()

    ptr_ref = Ref{Ptr{Cvoid}}()
    cuMemHostAlloc(ptr_ref, bytesize, flags)

    return HostMemory(context(), ptr_ref[], bytesize)
end


const MEMHOSTREGISTER_PORTABLE = CU_MEMHOSTREGISTER_PORTABLE
const MEMHOSTREGISTER_DEVICEMAP = CU_MEMHOSTREGISTER_DEVICEMAP
const MEMHOSTREGISTER_IOMEMORY = CU_MEMHOSTREGISTER_IOMEMORY

"""
    register(HostMemory, ptr::Ptr, bytesize::Integer, [flags])

Page-lock the host memory pointed to by `ptr`. Subsequent transfers to and from devices will
be faster, and can be executed asynchronously. If the `MEMHOSTREGISTER_DEVICEMAP` flag is
specified, the buffer will also be accessible directly from the GPU. These accesses are
direct, and go through the PCI bus. If the `MEMHOSTREGISTER_PORTABLE` flag is specified, any
CUDA context can access the memory.
"""
function register(::Type{HostMemory}, ptr::Ptr, bytesize::Integer, flags=0)
    bytesize == 0 && throw(ArgumentError("Cannot register an empty range of memory."))

    cuMemHostRegister_v2(ptr, bytesize, flags)

    return HostMemory(context(), ptr, bytesize)
end

"""
    unregister(::HostMemory)

Unregisters a memory range that was registered with [`register`](@ref).
"""
function unregister(mem::HostMemory)
    cuMemHostUnregister(mem)
end


function free(mem::HostMemory)
    if pointer(mem) != CU_NULL
        cuMemFreeHost(mem)
    end
end


## unified memory

"""
    UnifiedMemory

Unified memory that is accessible on both the CPU and GPU.
"""
struct UnifiedMemory <: AbstractMemory
    ctx::CuContext
    ptr::CuPtr{Cvoid}
    bytesize::Int
end

UnifiedMemory() = UnifiedMemory(context(), CU_NULL, 0)

Base.pointer(mem::UnifiedMemory) = mem.ptr
Base.sizeof(mem::UnifiedMemory) = mem.bytesize

Base.show(io::IO, mem::UnifiedMemory) =
    @printf(io, "UnifiedMemory(%s at %p)", Base.format_bytes(sizeof(mem)), Int(pointer(mem)))

Base.convert(::Type{Ptr{T}}, mem::UnifiedMemory) where {T} =
    convert(Ptr{T}, reinterpret(Ptr{Cvoid}, pointer(mem)))

Base.convert(::Type{CuPtr{T}}, mem::UnifiedMemory) where {T} =
    convert(CuPtr{T}, pointer(mem))

@enum_without_prefix CUmemAttach_flags CU_

"""
    alloc(UnifiedMemory, bytesize::Integer, [flags::CUmemAttach_flags])

Allocate `bytesize` bytes of unified memory. This memory is accessible from both the CPU and
GPU, with the CUDA driver automatically copying upon first access.
"""
function alloc(::Type{UnifiedMemory}, bytesize::Integer,
              flags::CUmemAttach_flags=MEM_ATTACH_GLOBAL)
    bytesize == 0 && return UnifiedMemory()

    ptr_ref = Ref{CuPtr{Cvoid}}()
    cuMemAllocManaged(ptr_ref, bytesize, flags)

    return UnifiedMemory(context(), ptr_ref[], bytesize)
end


function free(mem::UnifiedMemory)
    if pointer(mem) != CU_NULL
        cuMemFree_v2(mem)
    end
end


"""
    prefetch(::UnifiedMemory, [bytes::Integer]; [device::CuDevice], [stream::CuStream])

Prefetches memory to the specified destination device.
"""
function prefetch(mem::UnifiedMemory, bytes::Integer=sizeof(mem);
                  device::CuDevice=device(), stream::CuStream=stream())
    bytes > sizeof(mem) && throw(BoundsError(mem, bytes))
    cuMemPrefetchAsync(mem, bytes, device, stream)
end


@enum_without_prefix CUmem_advise CU_

"""
    advise(::UnifiedMemory, advice::CUDA.CUmem_advise, [bytes::Integer]; [device::CuDevice])

Advise about the usage of a given memory range.
"""
function advise(mem::UnifiedMemory, advice::CUmem_advise, bytes::Integer=sizeof(mem);
                device::CuDevice=device())
    bytes > sizeof(mem) && throw(BoundsError(mem, bytes))
    cuMemAdvise(mem, bytes, advice, device)
end


## array memory

"""
    ArrayMemory

Array memory residing on the GPU, possibly in a specially-formatted way.
"""
mutable struct ArrayMemory{T,N} <: AbstractMemory
    ctx::CuContext
    ptr::CuArrayPtr{T}
    dims::Dims{N}
end

Base.pointer(mem::ArrayMemory) = mem.ptr
Base.sizeof(mem::ArrayMemory) = error("Opaque array memory does not have a definite size")
Base.size(mem::ArrayMemory) = mem.dims
Base.length(mem::ArrayMemory) = prod(mem.dims)
Base.ndims(mem::ArrayMemory{<:Any,N}) where {N} = N

Base.show(io::IO, mem::ArrayMemory{T,1}) where {T} =
    @printf(io, "%g-element ArrayMemory{%s,%g}(%p)", length(mem), string(T), 1, Int(pointer(mem)))
Base.show(io::IO, mem::ArrayMemory{T}) where {T} =
    @printf(io, "%s ArrayMemory{%s,%g}(%p)", Base.inds2string(size(mem)), string(T), ndims(mem), Int(pointer(mem)))

# array memory is typed, so refuse arbitrary conversions
Base.convert(::Type{CuArrayPtr{T}}, mem::ArrayMemory{T}) where {T} =
    convert(CuArrayPtr{T}, pointer(mem))
# ... except for CuArrayPtr{Nothing}, which is used to call untyped API functions
Base.convert(::Type{CuArrayPtr{Nothing}}, mem::ArrayMemory)  =
    convert(CuArrayPtr{Nothing}, pointer(mem))

"""
    alloc(ArrayMemory, dims::Dims)

Allocate array memory with dimensions `dims`. The memory is accessible on the GPU, but
can only be used in conjunction with special intrinsics (e.g., texture intrinsics).
"""
function alloc(::Type{<:ArrayMemory{T}}, dims::Dims{N}) where {T,N}
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

    allocateArray_ref = Ref(CUDA_ARRAY3D_DESCRIPTOR(
        width, # Width::Csize_t
        height, # Height::Csize_t
        depth, # Depth::Csize_t
        format, # Format::CUarray_format
        UInt32(nchans(T)), # NumChannels::UInt32
        0))

    handle_ref = Ref{CUarray}()
    cuArray3DCreate_v2(handle_ref, allocateArray_ref)
    ptr = reinterpret(CuArrayPtr{T}, handle_ref[])

    return ArrayMemory{T,N}(context(), ptr, dims)
end

function free(mem::ArrayMemory)
    cuArrayDestroy(mem)
end

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



#
# operations on pointers
#

## initialization

"""
    memset(mem::CuPtr, value::Union{UInt8,UInt16,UInt32}, len::Integer; [stream::CuStream])

Initialize device memory by copying `val` for `len` times.
"""
memset

for T in [UInt8, UInt16, UInt32]
    bits = 8*sizeof(T)
    fn = Symbol("cuMemsetD$(bits)Async")
    @eval function memset(ptr::CuPtr{$T}, value::$T, len::Integer; stream::CuStream=stream())
        $(getproperty(CUDA, fn))(ptr, value, len, stream)
        return
    end
end


## copy operations

# XXX: also provide low-level memcpy?

for (fn, srcPtrTy, dstPtrTy) in (("cuMemcpyDtoHAsync_v2", :CuPtr, :Ptr),
                                 ("cuMemcpyHtoDAsync_v2", :Ptr,   :CuPtr),
                                 )
    @eval function Base.unsafe_copyto!(dst::$dstPtrTy{T}, src::$srcPtrTy{T}, N::Integer;
                                       stream::CuStream=stream(),
                                       async::Bool=false) where T
        $(getproperty(CUDA, Symbol(fn)))(dst, src, N*sizeof(T), stream)
        async || synchronize(stream)
        return dst
    end
end

function Base.unsafe_copyto!(dst::CuPtr{T}, src::CuPtr{T}, N::Integer;
                             stream::CuStream=stream(),
                             async::Bool=false) where T
    dst_dev = device(dst)
    src_dev = device(src)
    if dst_dev == src_dev
        cuMemcpyDtoDAsync_v2(dst, src, N*sizeof(T), stream)
    else
        cuMemcpyPeerAsync(dst, context(dst_dev),
                          src, context(src_dev),
                          N*sizeof(T), stream)
    end
    async || synchronize(stream)
    return dst
end

function Base.unsafe_copyto!(dst::CuArrayPtr{T}, doffs::Integer, src::Ptr{T}, N::Integer;
                             stream::CuStream=stream(),
                             async::Bool=false) where T
    cuMemcpyHtoAAsync_v2(dst, doffs, src, N*sizeof(T), stream)
    async || synchronize(stream)
    return dst
end

function Base.unsafe_copyto!(dst::Ptr{T}, src::CuArrayPtr{T}, soffs::Integer, N::Integer;
                             stream::CuStream=stream(),
                             async::Bool=false) where T
    cuMemcpyAtoHAsync_v2(dst, src, soffs, N*sizeof(T), stream)
    async || synchronize(stream)
    return dst
end

Base.unsafe_copyto!(dst::CuArrayPtr{T}, doffs::Integer, src::CuPtr{T}, N::Integer) where {T} =
    cuMemcpyDtoA_v2(dst, doffs, src, N*sizeof(T))

Base.unsafe_copyto!(dst::CuPtr{T}, src::CuArrayPtr{T}, soffs::Integer, N::Integer) where {T} =
    cuMemcpyAtoD_v2(dst, src, soffs, N*sizeof(T))

Base.unsafe_copyto!(dst::CuArrayPtr, src, N::Integer; kwargs...) =
    Base.unsafe_copyto!(dst, 0, src, N; kwargs...)

Base.unsafe_copyto!(dst, src::CuArrayPtr, N::Integer; kwargs...) =
    Base.unsafe_copyto!(dst, src, 0, N; kwargs...)

"""
    unsafe_copy2d!(dst, dstTyp, src, srcTyp, width, height=1;
                   dstPos=(1,1), dstPitch=0,
                   srcPos=(1,1), srcPitch=0,
                   async=false, stream=nothing)

Perform a 2D memory copy between pointers `src` and `dst`, at respectively position `srcPos`
and `dstPos` (1-indexed). Pitch can be specified for both the source and destination;
consult the CUDA documentation for more details. This call is executed asynchronously if
`async` is set, otherwise `stream` is synchronized.
"""
function unsafe_copy2d!(dst::Union{Ptr{T},CuPtr{T},CuArrayPtr{T}}, dstTyp::Type{<:AbstractMemory},
                        src::Union{Ptr{T},CuPtr{T},CuArrayPtr{T}}, srcTyp::Type{<:AbstractMemory},
                        width::Integer, height::Integer=1;
                        dstPos::CuDim=(1,1), dstPitch::Integer=0,
                        srcPos::CuDim=(1,1), srcPitch::Integer=0,
                        async::Bool=false, stream::CuStream=CUDA.stream()) where T
    srcPos = CuDim3(srcPos)
    @assert srcPos.z == 1
    dstPos = CuDim3(dstPos)
    @assert dstPos.z == 1

    srcMemoryType, srcHost, srcDevice, srcArray = if srcTyp == HostMemory
        CU_MEMORYTYPE_HOST,
        src::Ptr,
        0,
        0
    elseif srcTyp == DeviceMemory
        CU_MEMORYTYPE_DEVICE,
        0,
        src::CuPtr,
        0
    elseif srcTyp == UnifiedMemory
        CU_MEMORYTYPE_UNIFIED,
        0,
        reinterpret(CuPtr{Cvoid}, src),
        0
    elseif srcTyp == ArrayMemory
        CU_MEMORYTYPE_ARRAY,
        0,
        0,
        src::CuArrayPtr
    end

    dstMemoryType, dstHost, dstDevice, dstArray = if dstTyp == HostMemory
        CU_MEMORYTYPE_HOST,
        dst::Ptr,
        0,
        0
    elseif dstTyp == DeviceMemory
        CU_MEMORYTYPE_DEVICE,
        0,
        dst::CuPtr,
        0
    elseif dstTyp == UnifiedMemory
        CU_MEMORYTYPE_UNIFIED,
        0,
        reinterpret(CuPtr{Cvoid}, dst),
        0
    elseif dstTyp == ArrayMemory
        CU_MEMORYTYPE_ARRAY,
        0,
        0,
        dst::CuArrayPtr
    end

    params_ref = Ref(CUDA_MEMCPY2D(
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
    cuMemcpy2DAsync_v2(params_ref, stream)
    async || synchronize(stream)
    return dst
end

"""
    unsafe_copy3d!(dst, dstTyp, src, srcTyp, width, height=1, depth=1;
                   dstPos=(1,1,1), dstPitch=0, dstHeight=0,
                   srcPos=(1,1,1), srcPitch=0, srcHeight=0,
                   async=false, stream=nothing)

Perform a 3D memory copy between pointers `src` and `dst`, at respectively position `srcPos`
and `dstPos` (1-indexed). Both pitch and height can be specified for both the source and
destination; consult the CUDA documentation for more details. This call is executed
asynchronously if `async` is set, otherwise `stream` is synchronized.
"""
function unsafe_copy3d!(dst::Union{Ptr{T},CuPtr{T},CuArrayPtr{T}}, dstTyp::Type{<:AbstractMemory},
                        src::Union{Ptr{T},CuPtr{T},CuArrayPtr{T}}, srcTyp::Type{<:AbstractMemory},
                        width::Integer, height::Integer=1, depth::Integer=1;
                        dstPos::CuDim=(1,1,1), srcPos::CuDim=(1,1,1),
                        dstPitch::Integer=0, dstHeight::Integer=0,
                        srcPitch::Integer=0, srcHeight::Integer=0,
                        async::Bool=false, stream::CuStream=stream()) where T
    srcPos = CuDim3(srcPos)
    dstPos = CuDim3(dstPos)

    # JuliaGPU/CUDA.jl#863: cuMemcpy3DAsync calculates wrong offset
    #                       when using the stream-ordered memory allocator
    # NOTE: we apply the workaround unconditionally, since we want to keep this call cheap.
    if v"11.2" <= driver_version() <= v"11.3" #&& pools[device()].stream_ordered
        srcOffset = (srcPos.x-1)*sizeof(T) + srcPitch*((srcPos.y-1) + srcHeight*(srcPos.z-1))
        dstOffset = (dstPos.x-1)*sizeof(T) + dstPitch*((dstPos.y-1) + dstHeight*(dstPos.z-1))
    else
        srcOffset = 0
        dstOffset = 0
    end

    srcMemoryType, srcHost, srcDevice, srcArray = if srcTyp == HostMemory
        CU_MEMORYTYPE_HOST,
        src::Ptr + srcOffset,
        0,
        0
    elseif srcTyp == DeviceMemory
        CU_MEMORYTYPE_DEVICE,
        0,
        src::CuPtr + srcOffset,
        0
    elseif srcTyp == UnifiedMemory
        CU_MEMORYTYPE_UNIFIED,
        0,
        reinterpret(CuPtr{Cvoid}, src) + srcOffset,
        0
    elseif srcTyp == ArrayMemory
        CU_MEMORYTYPE_ARRAY,
        0,
        0,
        src::CuArrayPtr + srcOffset
    end

    dstMemoryType, dstHost, dstDevice, dstArray = if dstTyp == HostMemory
        CU_MEMORYTYPE_HOST,
        dst::Ptr + dstOffset,
        0,
        0
    elseif dstTyp == DeviceMemory
        CU_MEMORYTYPE_DEVICE,
        0,
        dst::CuPtr + dstOffset,
        0
    elseif dstTyp == UnifiedMemory
        CU_MEMORYTYPE_UNIFIED,
        0,
        reinterpret(CuPtr{Cvoid}, dst) + dstOffset,
        0
    elseif dstTyp == ArrayMemory
        CU_MEMORYTYPE_ARRAY,
        0,
        0,
        dst::CuArrayPtr + dstOffset
    end

    params_ref = Ref(CUDA_MEMCPY3D(
        # source
        srcOffset==0 ? (srcPos.x-1)*sizeof(T) : 0,
        srcOffset==0 ? srcPos.y-1             : 0,
        srcOffset==0 ? srcPos.z-1             : 0,
        0, # LOD
        srcMemoryType, srcHost, srcDevice, srcArray,
        C_NULL, # reserved
        srcPitch, srcHeight,
        # destination
        dstOffset==0 ? (dstPos.x-1)*sizeof(T) : 0,
        dstOffset==0 ? dstPos.y-1             : 0,
        dstOffset==0 ? dstPos.z-1             : 0,
        0, # LOD
        dstMemoryType, dstHost, dstDevice, dstArray,
        C_NULL, # reserved
        dstPitch, dstHeight,
        # extent
        width*sizeof(T), height, depth
    ))
    cuMemcpy3DAsync_v2(params_ref, stream)
    async || synchronize(stream)
    return dst
end



#
# auxiliary functionality
#

# given object, find base allocation
# pin that, or increase refcount
# finalizer, drop refcount, free if 0

## memory pinning

const __pin_lock = ReentrantLock()

# - IdDict does not free the memory
# - WeakRef dict does not unique the key by objectid
const __pinned_objects = Dict{Tuple{CuContext,Ptr{Cvoid}}, WeakRef}()

function pin(a::AbstractArray)
    ctx = context()
    ptr = pointer(a)

    Base.@lock __pin_lock begin
        # only pin an object once per context
        key = (ctx, convert(Ptr{Nothing}, ptr))
        if haskey(__pinned_objects, key) && __pinned_objects[key].value !== nothing
            return nothing
        end
        __pinned_objects[key] = WeakRef(a)
    end

     __pin(ptr, sizeof(a))
    finalizer(a) do _
        __unpin(ptr, ctx)
    end

    a
end

function pin(ref::Base.RefValue{T}) where T
    ctx = context()
    ptr = Base.unsafe_convert(Ptr{T}, ref)

    __pin(ptr, sizeof(T))
    finalizer(ref) do _
        __unpin(ptr, ctx)
    end

    ref
end

# derived arrays should always pin the parent memory range, because we may end up copying
# from or to that parent range (containing the derived range), and partially-pinned ranges
# are not supported:
#
# > Memory regions requested must be either entirely registered with CUDA, or in the case
# > of host pageable transfers, not registered at all. Memory regions spanning over
# > allocations that are both registered and not registered with CUDA are not supported and
# > will return CUDA_ERROR_INVALID_VALUE.
__pin(a::Union{SubArray, Base.ReinterpretArray, Base.ReshapedArray}) = __pin(parent(a))

# refcount the pinning per context, since we can only pin a memory range once
const __pinned_memory = Dict{Tuple{CuContext,Ptr{Cvoid}}, HostMemory}()
const __pin_count = Dict{Tuple{CuContext,Ptr{Cvoid}}, Int}()
function __pin(ptr::Ptr, sz::Int)
    ctx = context()
    key = (ctx, convert(Ptr{Nothing}, ptr))

    Base.@lock __pin_lock begin
        pin_count = if haskey(__pin_count, key)
            __pin_count[key] += 1
        else
            __pin_count[key] = 1
        end

        if pin_count == 1
            mem = register(HostMemory, ptr, sz)
            __pinned_memory[key] = mem
        elseif Base.JLOptions().debug_level >= 2
            # make sure we're pinning the exact same range
            @assert haskey(__pinned_memory, key) "Cannot find memory for $ptr with pin count $pin_count."
            mem = __pinned_memory[key]
            @assert sz == sizeof(mem) "Mismatch between pin request of $ptr: $sz vs. $(sizeof(mem))."
        end
    end

    return
end
function __unpin(ptr::Ptr, ctx::CuContext)
    key = (ctx, convert(Ptr{Nothing}, ptr))

    Base.@lock __pin_lock begin
        @assert haskey(__pin_count, key) "Cannot unpin unmanaged pointer $ptr."
        pin_count = __pin_count[key] -= 1

        if pin_count == 0
            mem = @inbounds __pinned_memory[key]
            context!(ctx; skip_destroyed=true) do
                unregister(mem)
            end
            delete!(__pinned_memory, key)
        end
    end

    return
end
function __pinned(ptr::Ptr, ctx::CuContext)
    key = (ctx, convert(Ptr{Nothing}, ptr))
    Base.@lock __pin_lock begin
        haskey(__pin_count, key)
    end
end


## pointer attributes

# TODO: iterable struct

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

"""
    context(ptr)

Identify the context memory was allocated in.
"""
context(ptr::Union{Ptr,CuPtr}) =
    CuContext(attribute(CUcontext, ptr, POINTER_ATTRIBUTE_CONTEXT))

"""
    device(ptr)

Identify the device memory was allocated on.
"""
device(x::Union{Ptr,CuPtr}) =
    CuDevice(convert(Int, attribute(Cuint, x, POINTER_ATTRIBUTE_DEVICE_ORDINAL)))

@enum_without_prefix CUmemorytype CU_
memory_type(x) = CUmemorytype(attribute(Cuint, x, POINTER_ATTRIBUTE_MEMORY_TYPE))

is_managed(x) = convert(Bool, attribute(Cuint, x, POINTER_ATTRIBUTE_IS_MANAGED))

"""
    host_pointer(ptr::CuPtr)

Returns the host pointer value through which `ptr`` may be accessed by by the
host program.
"""
host_pointer(x::CuPtr{T}) where {T} =
    attribute(Ptr{T}, x, POINTER_ATTRIBUTE_HOST_POINTER)

"""
    device_pointer(ptr::Ptr)

Returns the device pointer value through which `ptr` may be accessed by kernels
running in the current context.
"""
device_pointer(x::Ptr{T}) where {T} =
    attribute(CuPtr{T}, x, POINTER_ATTRIBUTE_HOST_POINTER)

function is_pinned(ptr::Ptr)
    # unpinned memory makes cuPointerGetAttribute return ERROR_INVALID_VALUE; but instead of
    # calling `memory_type` with an expensive try/catch we perform low-level API calls.
    ptr = reinterpret(CuPtr{Nothing}, ptr)
    data_ref = Ref{Cuint}()
    res = unchecked_cuPointerGetAttribute(data_ref, POINTER_ATTRIBUTE_MEMORY_TYPE, ptr)
    if res == ERROR_INVALID_VALUE
        false
    elseif res == SUCCESS
        data_ref[] == CU_MEMORYTYPE_HOST
    else
        throw_api_error(res)
    end
end



#
# other
#

## memory info

function memory_info()
    free_ref = Ref{Csize_t}()
    total_ref = Ref{Csize_t}()
    cuMemGetInfo_v2(free_ref, total_ref)
    return convert(Int, free_ref[]), convert(Int, total_ref[])
end

"""
    free_memory()

Returns the free amount of memory (in bytes), available for allocation by the CUDA context.
"""
free_memory() = Int(memory_info()[1])

"""
    total_memory()

Returns the total amount of memory (in bytes), available for allocation by the CUDA context.
"""
total_memory() = Int(memory_info()[2])
