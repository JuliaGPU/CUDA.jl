## formats

export cuda_texture_alias_type

const _type_to_cuarrayformat_dict = Dict{DataType,CUarray_format}(
    UInt8   => CU_AD_FORMAT_UNSIGNED_INT8,
    UInt16  => CU_AD_FORMAT_UNSIGNED_INT16,
    UInt32  => CU_AD_FORMAT_UNSIGNED_INT32,
    Int8    => CU_AD_FORMAT_SIGNED_INT8,
    Int16   => CU_AD_FORMAT_SIGNED_INT16,
    Int32   => CU_AD_FORMAT_SIGNED_INT32,
    Float16 => CU_AD_FORMAT_HALF,
    Float32 => CU_AD_FORMAT_FLOAT,
)

@inline _alias_type_to_nchan_and_format(::Type{T}) where {T} =
    error("Type `$T` is not a valid alias type for a \"CUDA array\" (texture memory) format")

for (T, cuarrayformat) in _type_to_cuarrayformat_dict
    @eval @inline _alias_type_to_nchan_and_format(::Type{$T})  = 1, $cuarrayformat, $T
    @eval @inline _alias_type_to_nchan_and_format(::Type{NTuple{2,$T}}) = 2, $cuarrayformat, $T
    @eval @inline _alias_type_to_nchan_and_format(::Type{NTuple{4,$T}}) = 4, $cuarrayformat, $T
end

@inline _assert_alias_size(::Type{T}, ::Type{Ta}) where {T,Ta} =
    sizeof(T) == sizeof(Ta) || throw(DimensionMismatch("Julia type `$T` cannot be aliased to the \"CUDA array\" (texture memory) format `$Ta`: sizes in bytes do not match"))

for (T, _) in _type_to_cuarrayformat_dict
    @eval @inline cuda_texture_alias_type(::Type{$T}) = $T
    for N in (2, 4)
        @eval @inline cuda_texture_alias_type(::Type{NT}) where {NT <: NTuple{$N,$T}} = NT
    end
end

@generated function cuda_texture_alias_type(t::Type{T}) where T
    err = "An alias from the type `$T` to a \"CUDA array\" (texture memory) format could not be inferred"
    isprimitivetype(T) &&
        return :(@error $("Primitive type `$T` does not have a defined alias to a \"CUDA array\" (texture memory) format."))
    isbitstype(T) || return :(@error $("$err: Type is not `isbitstype`."))
    Te = nothing
    N = 0
    datatypes = DataType[T]
    while !isempty(datatypes)
        for Ti in fieldtypes(pop!(datatypes))
            if isprimitivetype(Ti)
                Ti = cuda_texture_alias_type(Ti)
                typeof(Ti) == DataType ||
                    return :(@error $("$err: Composed of primitive type with no alias."))
                if !isprimitivetype(Ti)
                    push!(datatypes, Ti)
                    break
                end
                if Te == nothing
                    Te = Ti
                    N = 1
                elseif Te == Ti
                    N += 1
                else
                    return :(@error $("$err: Incompatible elements."))
                end
            else
                push!(datatypes, Ti)
            end
        end
    end
    Ta = NTuple{N,Te}
    sizeof(T) == sizeof(Ta) ||
        return :(@error $("$err: Inferred alias type and original type have different byte sizes."))
    in(N, (1, 2, 4)) ||
        return :(@error $("$err: Incompatible number of elements ($N, but only 1, 2 or 4 supported)."))
    return Ta
end


## array

export CuTextureArray

"""
    CuTextureArray{T,N}

Type to handle CUDA arrays which are opaque device memory buffers optimized for texture
fetching. The only way to initialize the content of this objects is by copying from host or
device arrays using the constructor or `copyto!` calls.
"""
mutable struct CuTextureArray{T,N}
    handle::CUarray
    dims::Dims{N}

    ctx::CuContext

    function CuTextureArray{T,N}(dims::Dims{N}) where {T,N}
        Ta = cuda_texture_alias_type(T)
        _assert_alias_size(T, Ta)
        nchan, format = _alias_type_to_nchan_and_format(Ta)

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
            UInt32(nchan), # NumChannels::UInt32
            0))

        handle_ref = Ref{CUarray}()
        cuArray3DCreate(handle_ref, allocateArray_ref)

        t = new{T,N}(handle_ref[], dims, context())
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

CuTextureArray{T}(n::Int) where {T} = CuTextureArray{T,1}((n,))
CuTextureArray{T}(nx::Int, ny::Int) where {T} = CuTextureArray{T,2}((nx, ny))
CuTextureArray{T}(nx::Int, ny::Int, nz::Int) where {T} = CuTextureArray{T,3}((nx, ny, nz))

Base.eltype(tm::CuTextureArray{T,N}) where {T,N} = T
Base.size(tm::CuTextureArray) = tm.dims
Base.length(tm::CuTextureArray) = prod(size(tm))
Base.sizeof(tm::CuTextureArray) = sizeof(eltype(tm)) * length(tm)


### Memory transfer

function Base.copyto!(dst::CuTextureArray{T,1}, src::Array{T,1}) where {T}
    size(dst) == size(src) || throw(DimensionMismatch("source and destination sizes must match"))
    cuMemcpyHtoA(dst, 0, src, sizeof(dst))
    return dst
end

function Base.copyto!(dst::CuTextureArray{T,1}, src::CuArray{T,1}) where {T}
    size(dst) == size(src) || throw(DimensionMismatch("source and destination sizes must match"))
    cuMemcpyDtoA(dst, 0, src, sizeof(dst))
    return dst
end

function Base.copyto!(dst::CuTextureArray{T,2}, src::Union{Array{T,2}, CuArray{T,2}}) where {T}
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
        dst.dims[1] * sizeof(T), # WidthInBytes::Csize_t
        dst.dims[2], # Height::Csize_t
    ))
    cuMemcpy2D(copy_ref)
    return dst
end

function Base.copyto!(dst::CuTextureArray{T,3}, src::Union{Array{T,3}, CuArray{T,3}}) where {T}
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
        size(src, 1) * sizeof(T), # srcPitch::Csize_t
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
        dst.dims[1] * sizeof(T), # WidthInBytes::Csize_t
        dst.dims[2], # Height::Csize_t
        dst.dims[3], # Depth::Csize_t
    ))
    cuMemcpy3D(copy_ref)
    return dst
end

function CuTextureArray(a::Union{Array{T}, CuArray{T}}) where {T}
    t = CuTextureArray{T}(size(a)...)
    copyto!(t, a)
    return t
end


## texture

export CuTexture, mode_wrap, mode_clamp, mode_mirror, mode_border, mode_point, mode_linear

const AddressMode = CUaddress_mode_enum
const mode_wrap = CU_TR_ADDRESS_MODE_WRAP
const mode_clamp = CU_TR_ADDRESS_MODE_CLAMP
const mode_mirror = CU_TR_ADDRESS_MODE_MIRROR
const mode_border = CU_TR_ADDRESS_MODE_BORDER

const FilterMode = CUfilter_mode_enum
const mode_point = CU_TR_FILTER_MODE_POINT
const mode_linear = CU_TR_FILTER_MODE_LINEAR

"""
    CuTexture{T,N,Mem}

Type to handle CUDA texture objects. These objects do not hold data by themselves, but
instead are bound either to `CuTextureArray`s (CUDA arrays) or to `CuArray`s. (Note: For
correct wrapping `CuArray`s it is necessary the their memory is well aligned and strided
(good pitch). Currently, that is not being enforced.) Theses objects are meant to be used to
do texture fetchts inside kernels. When passed to kernels, `CuTexture` objects are
transformed into `CuDeviceTexture`s objects.
"""
mutable struct CuTexture{T,N,Mem}
    mem::Mem
    handle::CUtexObject

    ctx::CuContext

    function CuTexture{T,N,Mem}(texmemory::Mem,
                                address_modes::NTuple{N,AddressMode},
                                filter_mode::FilterMode) where {T,N,Mem}
        Ta = cuda_texture_alias_type(T)
        _assert_alias_size(T, Ta)
        nchan, format, Te = _alias_type_to_nchan_and_format(Ta)

        resDesc_ref = CUDA_RESOURCE_DESC(texmemory)

        address_modes = tuple(address_modes..., ntuple(_->mode_clamp, 3 - N)...)
        filter_mode = filter_mode
        flags = zero(CU_TRSF_NORMALIZED_COORDINATES)  # use absolute (non-normalized) coordinates
        flags = flags | (Te <: Integer ? CU_TRSF_READ_AS_INTEGER : zero(CU_TRSF_READ_AS_INTEGER))

        texDesc_ref = Ref(CUDA_TEXTURE_DESC(
            address_modes, # addressMode::NTuple{3, CUaddress_mode}
            filter_mode, # filterMode::CUfilter_mode
            flags, # flags::UInt32
            1, # maxAnisotropy::UInt32
            filter_mode, # mipmapFilterMode::CUfilter_mode
            0, # mipmapLevelBias::Cfloat
            0, # minMipmapLevelClamp::Cfloat
            0, # maxMipmapLevelClamp::Cfloat
            ntuple(_->Cfloat(zero(Te)), 4), # borderColor::NTuple{4, Cfloat}
            ntuple(_->Cint(0), 12)))

        texObject_ref = Ref{CUtexObject}(0)
        cuTexObjectCreate(texObject_ref, resDesc_ref, texDesc_ref, C_NULL)

        t = new{T,N,Mem}(texmemory, texObject_ref[], context())
        finalizer(unsafe_destroy!, t)
        return t
    end
end

function CUDA_RESOURCE_DESC(texarr::CuTextureArray{T,N}) where {T,N}
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

function CUDA_RESOURCE_DESC(arr::CuArray{T,N}) where {T,N}
    # TODO: take care of allowed pitches
    1 <= N <= 2 || throw(ArgumentError("Only 1 or 2D CuArray objects can be wrapped in a texture"))

    Ta = cuda_texture_alias_type(T)
    _assert_alias_size(T, Ta)
    nchan, format, Te = _alias_type_to_nchan_and_format(Ta)

    # FIXME: manual construction due to invalid padding (JuliaInterop/Clang.jl#238)
    resDesc_ref = Ref(((N == 1 ? CU_RESOURCE_TYPE_LINEAR : CU_RESOURCE_TYPE_PITCH2D), # resType::CUresourcetype
                        pointer(arr), # 1 x UInt64 (CUdeviceptr)
                        format, # 1/2 x UInt64 (CUarray_format)
                        UInt32(nchan), # 1/2 x UInt64
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

@inline function CuTexture{T,N,Mem}(texmemory::Mem;
                            address_mode::AddressMode = mode_clamp,
                            address_modes::NTuple{N,AddressMode} = ntuple(_->address_mode, N),
                            filter_mode::FilterMode = mode_linear
                            ) where {T,N,Mem}
    CuTexture{T,N,Mem}(texmemory, address_modes, filter_mode)
end

CuTexture(texarr::CuTextureArray{T,N}; kwargs...) where {T,N} =
    CuTexture{T,N,CuTextureArray{T,N}}(texarr; kwargs...)
CuTexture{T}(n::Int; kwargs...) where {T} =
    CuTexture(CuTextureArray{T,1}((n,)); kwargs...)
CuTexture{T}(nx::Int, ny::Int; kwargs...) where {T} =
    CuTexture(CuTextureArray{T,2}((nx, ny)); kwargs...)
CuTexture{T}(nx::Int, ny::Int, nz::Int; kwargs...) where {T} =
    CuTexture(CuTextureArray{T,3}((nx, ny, nz)); kwargs...)
CuTexture(cuarr::CuArray{T,N}; kwargs...) where {T,N} =
    CuTexture{T,N,CuArray{T,N}}(cuarr; kwargs...)


Base.eltype(tm::CuTexture{T,N}) where {T,N} = T
Base.size(tm::CuTexture) = size(tm.mem)

Adapt.adapt_storage(::Adaptor, t::CuTexture{T,N}) where {T,N} =
    CuDeviceTexture{T,N}(t.handle, size(t.mem))
