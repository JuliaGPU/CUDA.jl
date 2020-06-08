## formats

export alias_type

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


## array

export CuTextureArray

"""
    CuTextureArray{T,N,C}

Type to handle CUDA arrays which are opaque device memory buffers optimized for texture
fetching. The only way to initialize the content of this objects is by copying from host or
device arrays using the constructor or `copyto!` calls.
"""
mutable struct CuTextureArray{T,N,C}
    handle::CUarray
    dims::Dims{N}

    ctx::CuContext

    function CuTextureArray{T,N,C}(::UndefInitializer, dims::Dims{N}) where {T,N,C}
        format = convert(CUarray_format, T)

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
            UInt32(C), # NumChannels::UInt32
            0))

        handle_ref = Ref{CUarray}()
        cuArray3DCreate(handle_ref, allocateArray_ref)

        t = new{T,N,C}(handle_ref[], dims, context())
        finalizer(unsafe_destroy!, t)
        return t
    end
end

function unsafe_destroy!(t::CuTextureArray)
    if isvalid(t.ctx)
        cuArrayDestroy(t)
    end
end

Base.unsafe_convert(::Type{CUarray}, t::CuTextureArray) = t.handle


## array interface

Base.eltype(tm::CuTextureArray{T,N,1}) where {T,N} = T
Base.eltype(tm::CuTextureArray{T,N,C}) where {T,N,C} = NTuple{C,T}

Base.elsize(tm::CuTextureArray) = sizeof(eltype(tm))
Base.size(tm::CuTextureArray) = tm.dims
Base.length(tm::CuTextureArray) = prod(size(tm))
Base.sizeof(tm::CuTextureArray) = sizeof(eltype(tm)) * length(tm)


## interop with other arrays

packed_type(T,C) = C==1 ? T : NTuple{C,T}

@inline function CuTextureArray{T,N,C}(xs::AbstractArray{<:Any,N}) where {T,N,C}
  A = CuTextureArray{T,N,C}(undef, size(xs))
  copyto!(A, convert(Array{packed_type(T,C)}, xs))
  return A
end

@inline function CuTextureArray{T,N,C}(xs::CuArray{<:Any,N}) where {T,N,C}
  A = CuTextureArray{T,N,C}(undef, size(xs))
  copyto!(A, convert(CuArray{packed_type(T,C)}, xs))
  return A
end

# idempotency
CuTextureArray{T,N,C}(xs::CuTextureArray{T,N,C}) where {T,N,C} = xs

CuTextureArray(A::AbstractArray{T,N}) where {T,N} = CuTextureArray{T,N,1}(A)
CuTextureArray(A::AbstractArray{NTuple{C,T},N}) where {T,N,C} = CuTextureArray{T,N,C}(A)


## memory operations

function Base.copyto!(dst::CuTextureArray{T,1,C}, src::Array{<:Any,1}) where {T,C}
    eltype(src) === packed_type(T,C) ||
        throw(ArgumentError("Invalid source element type for $C-channel $T: expected $(packed_type(T,C)), got $(eltype(src))"))
    size(dst) == size(src) || throw(DimensionMismatch("source and destination sizes must match"))
    cuMemcpyHtoA(dst, 0, src, sizeof(dst))
    return dst
end

function Base.copyto!(dst::CuTextureArray{T,1,C}, src::CuArray{<:Any,1}) where {T,C}
    eltype(src) === packed_type(T,C) ||
        throw(ArgumentError("Invalid source element type for $C-channel $T: expected $(packed_type(T,C)), got $(eltype(src))"))
    size(dst) == size(src) || throw(DimensionMismatch("source and destination sizes must match"))
    cuMemcpyDtoA(dst, 0, src, sizeof(dst))
    return dst
end

function Base.copyto!(dst::CuTextureArray{T,2,C}, src::Union{Array{<:Any,2}, CuArray{<:Any,2}}) where {T,C}
    eltype(src) === packed_type(T,C) ||
        throw(ArgumentError("Invalid source element type for $C-channel $T: expected $(packed_type(T,C)), got $(eltype(src))"))
    size(dst) == size(src) || throw(DimensionMismatch("source and destination sizes must match"))
    isdevmem = isa(src, CuArray)
    copy_ref = Ref(CUDA_MEMCPY2D(
        0, # srcXInBytes::Csize_t
        0, # srcY::Csize_t
        isdevmem ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST, # srcMemoryType::CUmemorytype
        isdevmem ? 0 : pointer(src), # srcHost::Ptr{Cvoid}
        isdevmem ? pointer(src) : 0, # srcDevice::CUdeviceptr
        0, # srcArray::CUarray
        0, # srcPitch::Csize_t ### TODO: check why this cannot be `size(src.dims, 1) * sizeof(T)` as it should
        0, # dstXInBytes::Csize_t
        0, # dstY::Csize_t
        CU_MEMORYTYPE_ARRAY, # dstMemoryType::CUmemorytype
        0, # dstHost::Ptr{Cvoid}
        0, # dstDevice::CUdeviceptr
        dst.handle, # dstArray::CUarray
        0, # dstPitch::Csize_t
        dst.dims[1] * sizeof(eltype(dst)), # WidthInBytes::Csize_t
        dst.dims[2], # Height::Csize_t
    ))
    cuMemcpy2D(copy_ref)
    return dst
end

function Base.copyto!(dst::CuTextureArray{T,3,C}, src::Union{Array{<:Any,3}, CuArray{<:Any,3}}) where {T,C}
    eltype(src) === packed_type(T,C) ||
        throw(ArgumentError("Invalid source element type for $C-channel $T: expected $(packed_type(T,C)), got $(eltype(src))"))
    size(dst) == size(src) || throw(DimensionMismatch("source and destination sizes must match"))
    isdevmem = isa(src, CuArray)
    copy_ref = Ref(CUDA_MEMCPY3D(0, # srcXInBytes::Csize_t
        0, # srcY::Csize_t
        0, # srcZ::Csize_t
        0, # srcLOD::Csize_t
        isdevmem ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST, # srcMemoryType::CUmemorytype
        isdevmem ? 0 : pointer(src), # srcHost::Ptr{Cvoid}
        isdevmem ? pointer(src) : 0, # srcDevice::CUdeviceptr
        0, # srcArray::CUarray
        0, # reserved0::Ptr{Cvoid}
        size(src, 1) * sizeof(eltype(dst)), # srcPitch::Csize_t
        size(src, 2), # srcHeight::Csize_t
        0, # dstXInBytes::Csize_t
        0, # dstY::Csize_t
        0, # dstZ::Csize_t
        0, # dstLOD::Csize_t
        CU_MEMORYTYPE_ARRAY, # dstMemoryType::CUmemorytype
        0, # dstHost::Ptr{Cvoid}
        0, # dstDevice::CUdeviceptr
        dst.handle, # dstArray::CUarray
        0, # reserved1::Ptr{Cvoid}
        0, # dstPitch::Csize_t
        0, # dstHeight::Csize_t
        dst.dims[1] * sizeof(eltype(dst)), # WidthInBytes::Csize_t
        dst.dims[2], # Height::Csize_t
        dst.dims[3], # Depth::Csize_t
    ))
    cuMemcpy3D(copy_ref)
    return dst
end


## texture

export CuTexture, mode_wrap, mode_clamp, mode_mirror, mode_border, mode_point, mode_linear

const AddressMode = CUaddress_mode_enum
const mode_wrap = CU_TR_ADDRESS_MODE_WRAP
const mode_clamp = CU_TR_ADDRESS_MODE_CLAMP
const mode_mirror = CU_TR_ADDRESS_MODE_MIRROR
const mode_border = CU_TR_ADDRESS_MODE_BORDER

@enum_without_prefix CUaddress_mode_enum CU_TR_

const FilterMode = CUfilter_mode_enum
const mode_point = CU_TR_FILTER_MODE_POINT
const mode_linear = CU_TR_FILTER_MODE_LINEAR

@enum_without_prefix CUfilter_mode_enum CU_TR_

"""
    CuTexture{T,N,C,P}

Type to handle CUDA texture objects. These objects do not hold data by themselves, but
instead are bound either to `CuTextureArray`s (CUDA arrays) or to `CuArray`s. (Note: For
correct wrapping `CuArray`s it is necessary the their memory is well aligned and strided
(good pitch). Currently, that is not being enforced.) Theses objects are meant to be used to
do texture fetchts inside kernels. When passed to kernels, `CuTexture` objects are
transformed into `CuDeviceTexture`s objects.
"""
mutable struct CuTexture{T,N,C,P}
    mem::Union{CuArray{T,N}, CuTextureArray{T,N,C}}
    handle::CUtexObject

    normalized_coordinates::Bool

    ctx::CuContext

    function CuTexture{T,N,C,P}(texmemory::P;
                                address_mode::Union{AddressMode,NTuple{N,AddressMode}}=ntuple(_->ADDRESS_MODE_CLAMP,N),
                                filter_mode::FilterMode=FILTER_MODE_POINT,
                                normalized_coordinates::Bool=false) where {T,N,C,P}
        format = convert(CUarray_format, T)

        resDesc_ref = CUDA_RESOURCE_DESC(texmemory, C, format)

        flags = 0x0
        if normalized_coordinates
            flags |= CU_TRSF_NORMALIZED_COORDINATES
        end
        if T <: Integer
            flags |= CU_TRSF_READ_AS_INTEGER
        end

        # we always need 3 address modes
        address_mode = tuple(address_mode..., ntuple(_->ADDRESS_MODE_CLAMP, 3 - N)...)

        texDesc_ref = Ref(CUDA_TEXTURE_DESC(
            address_mode, # addressMode::NTuple{3, CUaddress_mode}
            filter_mode, # filterMode::CUfilter_mode
            flags, # flags::UInt32
            1, # maxAnisotropy::UInt32
            filter_mode, # mipmapFilterMode::CUfilter_mode
            0, # mipmapLevelBias::Cfloat
            0, # minMipmapLevelClamp::Cfloat
            0, # maxMipmapLevelClamp::Cfloat
            ntuple(_->Cfloat(zero(T)), 4), # borderColor::NTuple{4, Cfloat}
            ntuple(_->Cint(0), 12)))

        texObject_ref = Ref{CUtexObject}(0)
        cuTexObjectCreate(texObject_ref, resDesc_ref, texDesc_ref, C_NULL)

        t = new{T,N,C,P}(texmemory, texObject_ref[], normalized_coordinates, context())
        finalizer(unsafe_destroy!, t)
        return t
    end
end

function CUDA_RESOURCE_DESC(texarr::CuTextureArray{T,N}, channels, format) where {T,N}
    # FIXME: manual construction due to invalid padding (JuliaInterop/Clang.jl#238)
    # res = Ref{ANONYMOUS1_res}()
    # unsafe_store!(Ptr{CUarray}(pointer_from_objref(res)), texarr.handle)
    # resDesc_ref = Ref(CUDA_RESOURCE_DESC(
    #     CU_RESOURCE_TYPE_ARRAY, # resType::CUresourcetype
    #     res[], # res::ANONYMOUS1_res
    #     0 # flags::UInt32
    # ))
    resDesc_ref = Ref((CU_RESOURCE_TYPE_ARRAY, # resType::CUresourcetype
                       texarr.handle, # 1 x UInt64
                       ntuple(_->Int64(0), 15), # 15 x UInt64
                       UInt32(0)))
    return resDesc_ref
end

# allow passing mismatching refs where we're expecting a CUDA_RESOURCE_DESC
Base.unsafe_convert(::Type{Ptr{CUDA_RESOURCE_DESC}}, ref::Base.RefValue{T}) where {T} =
    convert(Ptr{CUDA_RESOURCE_DESC}, Base.unsafe_convert(Ptr{T}, ref))

function CUDA_RESOURCE_DESC(arr::CuArray{T,N}, channels, format) where {T,N}
    # TODO: take care of allowed pitches
    1 <= N <= 2 || throw(ArgumentError("Only 1 or 2D CuArray objects can be wrapped in a texture"))

    # FIXME: manual construction due to invalid padding (JuliaInterop/Clang.jl#238)
    resDesc_ref = Ref(((N == 1 ? CU_RESOURCE_TYPE_LINEAR : CU_RESOURCE_TYPE_PITCH2D), # resType::CUresourcetype
                       pointer(arr), # 1 x UInt64 (CUdeviceptr)
                       format, # 1/2 x UInt64 (CUarray_format)
                       UInt32(channels), # 1/2 x UInt64
                       (N == 2 ? size(arr, 1) : size(arr, 1) * sizeof(T)), # 1 x UInt64 nx
                       (N == 2 ? size(arr, 2) : 0), # 1 x UInt64 ny
                       (N == 2 ? size(arr, 1) * sizeof(T) : 0), # 1 x UInt64 pitch
                       ntuple(_->Int64(0), 11), # 11 x UInt64
                       UInt32(0)))
    return resDesc_ref
end

function unsafe_destroy!(t::CuTexture)
    if isvalid(t.ctx)
        cuTexObjectDestroy(t)
    end
end

Base.convert(::Type{CUtexObject}, t::CuTexture) = t.handle


## array interface

Base.eltype(tm::CuTexture{T,N,1}) where {T,N} = T
Base.eltype(tm::CuTexture{T,N,C}) where {T,N,C} = NTuple{C,T}

Base.size(tm::CuTexture) = size(tm.mem)


## interop with other arrays

CuTexture{T,N,C}(x::Union{CuArray{T,N},CuTextureArray{T,N,C}}; kwargs...) where {T,N,C} =
    CuTexture{T,N,C,typeof(x)}(x; kwargs...)

CuTexture(x::CuTextureArray{T,N,C}; kwargs...) where {T,N,C} =
    CuTexture{T,N,C}(x; kwargs...)
CuTexture(x::CuArray{T,N}; kwargs...) where {T,N} =
    CuTexture{T,N,1}(x; kwargs...)
CuTexture(x::CuArray{NTuple{C,T},N}; kwargs...) where {T,N,C} =
    CuTexture{T,N,C}(x; kwargs...)

Adapt.adapt_storage(::Adaptor, t::CuTexture{T,N,C}) where {T,N,C} =
    CuDeviceTexture{T,N,C,t.normalized_coordinates}(size(t.mem), t.handle)
