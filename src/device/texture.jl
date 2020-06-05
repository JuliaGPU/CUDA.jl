export CuDeviceTexture

"""
Lightweight type to handle CUDA texture objects inside kernels. Textures are fetched through
indexing operations on `CuTexture`/`CuDeviceTexture` objects, e.g., `cutexture2d[0.2f0,
0.2f0]`.
"""
struct CuDeviceTexture{T,N,NC}
    dims::Dims{N}
    handle::CUtexObject
end

Base.convert(::Type{Int64}, t::CuDeviceTexture) = reinterpret(Int64, t.handle)

tex1D(texObject::CuDeviceTexture, x::Float32) =
    ccall("llvm.nvvm.tex.unified.1d.v4s32.f32", llvmcall,
          NTuple{4,Int32}, (Int64, Float32), texObject, x)

tex2D(texObject::CuDeviceTexture, x::Float32, y::Float32) =
    ccall("llvm.nvvm.tex.unified.2d.v4s32.f32", llvmcall,
          NTuple{4,Int32}, (Int64, Float32, Float32), texObject, x, y)

tex3D(texObject::CuDeviceTexture, x::Float32, y::Float32, z::Float32) =
    ccall("llvm.nvvm.tex.unified.3d.v4s32.f32", llvmcall,
          NTuple{4,Int32}, (Int64, Float32, Float32, Float32), texObject, x, y, z)

@inline texXD(t::CuDeviceTexture{<:Any,1}, x::Real) = tex1D(t, x)
@inline texXD(t::CuDeviceTexture{<:Any,2}, x::Real, y::Real) = tex2D(t, x, y)
@inline texXD(t::CuDeviceTexture{<:Any,3}, x::Real, y::Real, z::Real) = tex3D(t, x, y, z)

@inline reconstruct(::Type{T}, x::Int32) where {T <: Union{Int32,UInt32,Int16,UInt16,Int8,UInt8}} = unsafe_trunc(T, x)
@inline reconstruct(::Type{Float32}, x::Int32) = reinterpret(Float32, x)
@inline reconstruct(::Type{Float16}, x::Int32) = convert(Float16, reinterpret(Float32, x))

@inline reconstruct(::Type{T}, i32_x4::NTuple{4,Int32}) where T = reconstruct(T, i32_x4[1])
@inline reconstruct(::Type{NTuple{2,T}}, i32_x4::NTuple{4,Int32}) where T = (reconstruct(T, i32_x4[1]),
                                                                             reconstruct(T, i32_x4[2]))
@inline reconstruct(::Type{NTuple{4,T}}, i32_x4::NTuple{4,Int32}) where T = (reconstruct(T, i32_x4[1]),
                                                                             reconstruct(T, i32_x4[2]),
                                                                             reconstruct(T, i32_x4[3]),
                                                                             reconstruct(T, i32_x4[4]))

@inline function cast(::Type{T}, x) where {T}
    @assert sizeof(T) == sizeof(x)
    r = Ref(x)
    GC.@preserve r begin
       unsafe_load(convert(Ptr{T}, Base.unsafe_convert(Ptr{Cvoid}, r)))
    end
end

@inline function Base.getindex(t::CuDeviceTexture{T,N,NC}, idx::Vararg{<:Real,N}) where {T,N,NC}
    i32_x4 = if NC
        # normalized coordinates range between 0 and 1, and can be used as-is
        texXD(t, idx...)
    else
        # non-normalized coordinates should be adjusted for 1-based indexing
        texXD(t, ntuple(i->idx[i]-1, N)...)
    end
    Ta = cuda_texture_alias_type(T)
    cast(T, reconstruct(Ta, i32_x4))
end
