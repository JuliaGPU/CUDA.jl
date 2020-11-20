export CuDeviceTexture

abstract type TextureInterpolationMode end
struct NearestNeighbour      <: TextureInterpolationMode end
struct LinearInterpolation   <: TextureInterpolationMode end
struct CubicInterpolation    <: TextureInterpolationMode end

abstract type TextureMemorySource end
struct ArrayMemory   <: TextureMemorySource end
struct LinearMemory  <: TextureMemorySource end

"""
    CuDeviceTexture{T,N,M,NC,I}

`N`-dimensional device texture with elements of type `T`. This type is the device-side
counterpart of [`CuTexture{T,N,P}`](@ref), and can be used to access textures using regular
indexing notation. If `NC` is true, indices used by these accesses should be normalized,
i.e., fall into the `[0,1)` domain. The `I` type parameter indicates the kind of
interpolation that happens when indexing into this texture. The source memory of the
texture is specified by the `M` parameter, either linear memory or a texture array.

Device-side texture objects cannot be created directly, but should be created host-side
using [`CuTexture{T,N,P}`](@ref) and passed to the kernal as an argument.

!!! warning
    Experimental API. Subject to change without deprecation.
"""
struct CuDeviceTexture{T,N,M<:TextureMemorySource,NC,I<:TextureInterpolationMode} <: AbstractArray{T,N}
    dims::Dims{N}
    handle::CUtexObject
end

Base.convert(::Type{CUtexObject}, t::CuDeviceTexture) = t.handle


## array interface

Base.elsize(::Type{<:CuDeviceTexture{T}}) where {T} = sizeof(T)

Base.size(tm::CuDeviceTexture) = tm.dims
Base.sizeof(tm::CuDeviceTexture) = Base.elsize(x) * length(x)


## low-level operations

# Source: NVVM IR specification 1.4
# NOTE: tex1Dfetch (integer coordinates) is unsupported, as it can be easily done using ldg

struct Vec4{T}
    a::T
    b::T
    c::T
    d::T
end

Base.Tuple(x::Vec4) = tuple(x.a, x.b, x.c, x.d)

for (dispatch_rettyp, julia_rettyp, llvm_rettyp) in
        ((Signed,        Vec4{UInt32},  :v4u32),
         (Unsigned,      Vec4{Int32},   :v4s32),
         (AbstractFloat, Vec4{Float32}, :v4f32))

    eltyp = Union{dispatch_rettyp, NTuple{<:Any,dispatch_rettyp}}

    # tex1D only supports array memory
    @eval tex(texObject::CuDeviceTexture{<:$eltyp,1,ArrayMemory}, x::Number) =
        Tuple(ccall($("llvm.nvvm.tex.unified.1d.$llvm_rettyp.f32"), llvmcall,
                    $julia_rettyp, (CUtexObject, Float32), texObject, x))

    # tex2D and tex3D supports all memories
    for dims in 2:3
        llvm_dim = "$(dims)d"
        julia_args = (:x, :y, :z)[1:dims]
        julia_sig = ntuple(_->Float32, dims)
        julia_params = ntuple(i->:($(julia_args[i])::Number), dims)

        @eval tex(texObject::CuDeviceTexture{<:$eltyp,$dims,}, $(julia_params...)) =
            Tuple(ccall($("llvm.nvvm.tex.unified.$llvm_dim.$llvm_rettyp.f32"), llvmcall,
                        $julia_rettyp, (CUtexObject, $(julia_sig...)), texObject, $(julia_args...)))
    end
end


## hardware-supported indexing

# we only support Float32 indices
@inline Base.getindex(t::CuDeviceTexture, idx::Vararg{<:Real,N}) where {N} =
    Base.getindex(t, ntuple(i->Float32(idx[i]), N)...)

@inline function Base.getindex(t::CuDeviceTexture{T,N,<:Any,true,I}, idx::Vararg{Float32,N}) where
                              {T,N,I<:Union{NearestNeighbour,LinearInterpolation}}
    # normalized coordinates range between 0 and 1, and can be used as-is
    vals = tex(t, idx...)
    return unpack(T, vals)
end

@inline function Base.getindex(t::CuDeviceTexture{T,N,<:Any,false,I}, idx::Vararg{Float32,N}) where
                              {T,N,I<:Union{NearestNeighbour,LinearInterpolation}}
    # non-normalized coordinates should be adjusted for 1-based indexing
    vals = tex(t, ntuple(i->idx[i]-0.5, N)...)
    return unpack(T, vals)
end

# unpack single-channel texture fetches as values, tuples otherwise
@inline unpack(::Type{T},           vals::NTuple) where T = unpack(T, vals[1])
@inline unpack(::Type{NTuple{1,T}}, vals::NTuple) where T = unpack(T, vals[1])
@inline unpack(::Type{NTuple{C,T}}, vals::NTuple) where {C,T} = ntuple(i->unpack(T, vals[i]), C)

@inline unpack(::Type{T}, val::T) where {T} = val
@inline unpack(::Type{T}, val::Real) where {T <: Integer} = unsafe_trunc(T, val)
@inline unpack(::Type{Float16}, val::Float32) = convert(Float16, val)


## cubic indexing (building on linear filtering)

# Source: GPU Gems 2, Chapter 20: Fast Third-Order Texture Filtering
#         CUDA sample: bicubicTextures

# cubic B-spline basis functions
w0(a::Float32) = (1.0f0/6.0f0)*(a*(a*(-a + 3.0f0) - 3.0f0) + 1.0f0)
w1(a::Float32) = (1.0f0/6.0f0)*(a*a*(3.0f0*a - 6.0f0) + 4.0f0)
w2(a::Float32) = (1.0f0/6.0f0)*(a*(a*(-3.0f0*a + 3.0f0) + 3.0f0) + 1.0f0)
w3(a::Float32) = (1.0f0/6.0f0)*(a*a*a)

# amplitude functions
g0(a::Float32) = w0(a) + w1(a)
g1(a::Float32) = w2(a) + w3(a)

# offset functions
# NOTE: +0.5 offset to compensate for CUDA linear filtering convention
h0(a::Float32) = -1.0f0 + w1(a) / (w0(a) + w1(a)) + 0.5f0
h1(a::Float32) = 1.0f0 + w3(a) / (w2(a) + w3(a)) + 0.5f0

@inline function Base.getindex(t::CuDeviceTexture{T,1,<:Any,false,CubicInterpolation},
                               x::Float32) where {T}
    x -= 1.0f0
    px = floor(x)   # integer position
    fx = x - px     # fractional position

    g0x = g0(fx)
    g1x = g1(fx)
    h0x = h0(fx)
    h1x = h1(fx)

    vals = g0x .* tex(t, px + h0x) .+ g1x .* tex(t, px + h1x)
    return (unpack(T, vals))
end

@inline function Base.getindex(t::CuDeviceTexture{T,2,<:Any,false,CubicInterpolation},
                               x::Float32, y::Float32) where {T}
    x -= 1.0f0
    y -= 1.0f0
    px = floor(x)   # integer position
    py = floor(y)
    fx = x - px     # fractional position
    fy = y - py

    g0x = g0(fx)
    g1x = g1(fx)
    h0x = h0(fx)
    h1x = h1(fx)
    h0y = h0(fy)
    h1y = h1(fy)

    vals = g0(fy) .* (g0x .* tex(t, px + h0x, py + h0y) .+
                      g1x .* tex(t, px + h1x, py + h0y)) .+
           g1(fy) .* (g0x .* tex(t, px + h0x, py + h1y) .+
                      g1x .* tex(t, px + h1x, py + h1y))
    return (unpack(T, vals))
end
