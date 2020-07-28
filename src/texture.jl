# texture and surface handling

# NOTE: the API for texture support is not final yet. some thoughts:
#
# - instead of CuTextureArray, use CuArray with an ArrayBuffer. This array could then
#   adapt to a CuTexture, or do the same for CuDeviceArray.

#
# Texture array
#

export CuTextureArray

"""
    CuTextureArray{T,N}(undef, dims)

`N`-dimensional dense texture array with elements of type `T`. These arrays are optimized
for texture fetching, and are only meant to be used as a source for [`CUDA.CuTexture`](@ref)
objects.

!!! warning
    Experimental API. Subject to change without deprecation.
"""
mutable struct CuTextureArray{T,N}
    buf::Mem.ArrayBuffer{T}
    dims::Dims{N}
    ctx::CuContext

    @doc """
        CuTextureArray{T,N}(undef, dims)

    Construct an uninitialized texture array of `N` dimensions specified in the `dims`
    tuple, with elements of type `T`. Use `Base.copyto!` to initialize this texture array,
    or use constructors that take a non-texture array to do so automatically.

    !!! warning
        Experimental API. Subject to change without deprecation.
    """
    function CuTextureArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N}
        buf = Mem.alloc(Mem.Array{T}, dims)
        t = new{T,N}(buf, dims, context())
        finalizer(unsafe_destroy!, t)
        return t
    end
end

function unsafe_destroy!(t::CuTextureArray)
    if isvalid(t.ctx)
        Mem.free(t.buf)
    end
end

Base.unsafe_convert(T::Type{CUarray}, t::CuTextureArray) = Base.unsafe_convert(T, t.buf)


## array interface

Base.size(tm::CuTextureArray) = tm.dims
Base.length(tm::CuTextureArray) = prod(size(tm))

Base.eltype(tm::CuTextureArray{T,N}) where {T,N} = T

Base.sizeof(tm::CuTextureArray) = sizeof(eltype(tm)) * length(tm)

Base.pointer(t::CuTextureArray) = t.buf.ptr


## interop with other arrays

"""
    CuTextureArray(A::AbstractArray)

Allocate and initialize a texture buffer from host memory in `A`.

!!! warning
    Experimental API. Subject to change without deprecation.
"""
@inline function CuTextureArray{T,N}(xs::AbstractArray{<:Any,N}) where {T,N}
  A = CuTextureArray{T,N}(undef, size(xs))
  copyto!(A, convert(Array{T}, xs))
  return A
end

"""
    CuTextureArray(A::CuArray)

Allocate and initialize a texture buffer from device memory in `A`.

!!! warning
    Experimental API. Subject to change without deprecation.
"""
@inline function CuTextureArray{T,N}(xs::CuArray{<:Any,N}) where {T,N}
  A = CuTextureArray{T,N}(undef, size(xs))
  copyto!(A, convert(CuArray{T}, xs))
  return A
end

# idempotency
CuTextureArray{T,N}(xs::CuTextureArray{T,N}) where {T,N} = xs

CuTextureArray(A::AbstractArray{T,N}) where {T,N} = CuTextureArray{T,N}(A)


## memory operations

function Base.copyto!(dst::CuTextureArray{T,1}, src::Union{Array{T,1}, CuArray{T,1}}) where {T}
    size(dst) == size(src) || throw(DimensionMismatch("source and destination sizes must match"))
    Base.unsafe_copyto!(pointer(dst), pointer(src), length(dst))
    return dst
end

function Base.copyto!(dst::CuTextureArray{T,2}, src::Union{Array{T,2}, CuArray{T,2}}) where {T}
    size(dst) == size(src) || throw(DimensionMismatch("source and destination sizes must match"))
    Mem.unsafe_copy2d!(pointer(dst), Mem.Array,
                       pointer(src), isa(src, Array) ? Mem.Host : Mem.Device,
                       size(dst)...)
    return dst
end

function Base.copyto!(dst::CuTextureArray{T,3}, src::Union{Array{T,3}, CuArray{T,3}}) where {T}
    size(dst) == size(src) || throw(DimensionMismatch("source and destination sizes must match"))
    Mem.unsafe_copy3d!(pointer(dst), Mem.Array,
                       pointer(src), isa(src, Array) ? Mem.Host : Mem.Device,
                       size(dst)...)
    return dst
end



#
# Texture objects
#

export CuTexture

@enum_without_prefix CUaddress_mode CU_TR_

@enum_without_prefix CUfilter_mode CU_TR_

"""
    CuTexture{T,N,P}

`N`-dimensional texture object with elements of type `T`. These objects do not store data
themselves, but are bounds to another source of device memory. Texture objects can be passed
to CUDA kernels, where they will be accessible through the [`CuDeviceTexture`](@ref) type.

!!! warning
    Experimental API. Subject to change without deprecation.
"""
mutable struct CuTexture{T,N,P} <: AbstractArray{T,N}
    parent::P
    handle::CUtexObject

    normalized_coordinates::Bool

    ctx::CuContext

    @doc """
        CuTexture{T,N,P}(parent::P; address_mode, filter_mode, normalized_coordinates)

    Construct a `N`-dimensional texture object with elements of type `T` as stored in
    `parent`.

    Several keyword arguments alter the behavior of texture objects:
    - `address_mode` (wrap, *clamp*, mirror): how out-of-bounds values are accessed. Can be
      specified as a value for all dimensions, or as a tuple of `N` entries.
    - `filter_mode` (*point*, linear): how non-integral indices are fetched. Point mode
      fetches a single value, linear results in linear interpolation between values.
    - `normalized_coordinates` (true, *false*): whether indices are expected to fall in the
      normalized `[0:1)` range.

    !!! warning
        Experimental API. Subject to change without deprecation.
    """
    function CuTexture{T,N,P}(parent::P;
                              address_mode::Union{CUaddress_mode,NTuple{N,CUaddress_mode}}=ntuple(_->ADDRESS_MODE_CLAMP,N),
                              filter_mode::CUfilter_mode=FILTER_MODE_POINT,
                              normalized_coordinates::Bool=false) where {T,N,P}
        resDesc_ref = CUDA_RESOURCE_DESC(parent)

        flags = 0x0
        if normalized_coordinates
            flags |= CU_TRSF_NORMALIZED_COORDINATES
        end
        if eltype(T) <: Integer
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
            ntuple(_->Cfloat(zero(eltype(T))), 4), # borderColor::NTuple{4, Cfloat}
            ntuple(_->Cint(0), 12)))

        texObject_ref = Ref{CUtexObject}(0)
        cuTexObjectCreate(texObject_ref, resDesc_ref, texDesc_ref, C_NULL)

        t = new{T,N,P}(parent, texObject_ref[], normalized_coordinates, context())
        finalizer(unsafe_destroy!, t)
        return t
    end
end

function CUDA_RESOURCE_DESC(texarr::CuTextureArray)
    # FIXME: manual construction due to invalid padding (JuliaInterop/Clang.jl#238)
    # res = Ref{ANONYMOUS1_res}()
    # unsafe_store!(Ptr{CUarray}(pointer_from_objref(res)), texarr.handle)
    # resDesc_ref = Ref(CUDA_RESOURCE_DESC(
    #     CU_RESOURCE_TYPE_ARRAY, # resType::CUresourcetype
    #     res[], # res::ANONYMOUS1_res
    #     0 # flags::UInt32
    # ))
    resDesc_ref = Ref((CU_RESOURCE_TYPE_ARRAY, # resType::CUresourcetype
                       pointer(texarr), # 1 x UInt64
                       ntuple(_->Int64(0), 15), # 15 x UInt64
                       UInt32(0)))
    return resDesc_ref
end

function CUDA_RESOURCE_DESC(arr::CuArray{T,N}) where {T,N}
    # TODO: take care of allowed pitches
    1 <= N <= 2 || throw(ArgumentError("Only 1 or 2D CuArray objects can be wrapped in a texture"))

    format = convert(CUarray_format, eltype(T))
    channels = nchans(T)

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

# allow passing mismatching refs where we're expecting a CUDA_RESOURCE_DESC
Base.unsafe_convert(::Type{Ptr{CUDA_RESOURCE_DESC}}, ref::Base.RefValue{T}) where {T} =
    convert(Ptr{CUDA_RESOURCE_DESC}, Base.unsafe_convert(Ptr{T}, ref))

function unsafe_destroy!(t::CuTexture)
    if isvalid(t.ctx)
        cuTexObjectDestroy(t)
    end
end

Base.convert(::Type{CUtexObject}, t::CuTexture) = t.handle

Base.parent(tm::CuTexture) = tm.parent


## array interface

Base.size(tm::CuTexture) = size(tm.parent)
Base.sizeof(tm::CuTexture) = Base.elsize(x) * length(x)

Base.show(io::IO, t::CuTexture{T,1}) where {T} =
    print(io, "$(length(t))-element $(nchans(T))-channel CuTexture(::$(typeof(parent(t)).name)) with eltype $T")
Base.show(io::IO, t::CuTexture{T}) where {T} =
    print(io, "$(join(size(t), '×')) $(nchans(T))-channel CuTexture(::$(typeof(parent(t)).name)) with eltype $T")

Base.show(io::IO, mime::MIME"text/plain", t::CuTexture) = show(io, t)


## interop with other arrays

CuTexture{T,N}(A::Union{CuTextureArray{T,N},CuArray{T,N}}; kwargs...) where {T,N} =
    CuTexture{T,N,typeof(A)}(A; kwargs...)

"""
    CuTexture(x::CuTextureArray{T,N})

Create a `N`-dimensional texture object withelements of type `T` that will be read from `x`.

!!! warning
    Experimental API. Subject to change without deprecation.
"""
CuTexture(x::CuTextureArray{T,N}; kwargs...) where {T,N} =
    CuTexture{T,N}(x; kwargs...)

"""
    CuTexture(x::CuArray{T,N})

Create a `N`-dimensional texture object that reads from a `CuArray`.

Note that it is necessary the their memory is well aligned and strided (good pitch).
Currently, that is not being enforced.

!!! warning
    Experimental API. Subject to change without deprecation.
"""
CuTexture(x::CuArray{T,N}; kwargs...) where {T,N} =
    CuTexture{T,N}(x; kwargs...)

Adapt.adapt_storage(::Adaptor, t::CuTexture{T,N}) where {T,N} =
    CuDeviceTexture{T,N,t.normalized_coordinates}(size(t), t.handle)
