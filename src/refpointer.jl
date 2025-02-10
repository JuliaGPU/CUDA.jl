# reference objects

abstract type AbstractCuRef{T} <: Ref{T} end

## opaque reference type
##
## we use a concrete CuRef type that actual references can be (no-op) converted to, without
## actually being a subtype of CuRef. This is necessary so that `CuRef` can be used in
## `ccall` signatures; which Base solves by special-casing `Ref` handing in `ccall.cpp`.
# forward declaration in pointer.jl

# general methods for CuRef{T} type
Base.eltype(x::Type{<:CuRef{T}}) where {T} = @isdefined(T) ? T : Any

Base.convert(::Type{CuRef{T}}, x::CuRef{T}) where {T} = x

# conversion or the actual ccall
Base.unsafe_convert(::Type{CuRef{T}}, x::CuRef{T}) where {T} = Base.bitcast(CuRef{T}, Base.unsafe_convert(CuPtr{T}, x))
Base.unsafe_convert(::Type{CuRef{T}}, x) where {T} = Base.bitcast(CuRef{T}, Base.unsafe_convert(CuPtr{T}, x))
## `@gcsafe_ccall` results in "double conversions" (remove this once `ccall` does `gcsafe`)
Base.unsafe_convert(::Type{CuPtr{T}}, x::CuRef{T}) where {T} = x

# CuRef from literal pointer
Base.convert(::Type{CuRef{T}}, x::CuPtr{T}) where {T} = x

# indirect constructors using CuRef
CuRef(x::Any) = CuRefValue(x)
CuRef{T}(x) where {T} = CuRefValue{T}(x)
CuRef{T}() where {T} = CuRefValue{T}()
Base.convert(::Type{CuRef{T}}, x) where {T} = CuRef{T}(x)

# idempotency
Base.convert(::Type{CuRef{T}}, x::AbstractCuRef{T}) where {T} = x


## reference backed by a single allocation

# TODO: maintain a small global cache of reference boxes

mutable struct CuRefValue{T} <: AbstractCuRef{T}
    buf::Managed{DeviceMemory}

    function CuRefValue{T}() where {T}
        check_eltype("CuRef", T)
        buf = pool_alloc(DeviceMemory, sizeof(T))
        obj = new(buf)
        finalizer(obj) do _
            pool_free(buf)
        end
        return obj
    end
end
function CuRefValue{T}(x::T) where {T}
    ref = CuRefValue{T}()
    ref[] = x
    return ref
end
CuRefValue{T}(x) where {T} = CuRefValue{T}(convert(T, x))
CuRefValue(x::T) where {T} = CuRefValue{T}(x)

Base.unsafe_convert(::Type{CuPtr{T}}, b::CuRefValue{T}) where {T} = convert(CuPtr{T}, b.buf)
Base.unsafe_convert(P::Type{CuPtr{Any}}, b::CuRefValue{Any}) = convert(P, b.buf)
Base.unsafe_convert(::Type{CuPtr{Cvoid}}, b::CuRefValue{T}) where {T} =
    convert(CuPtr{Cvoid}, b.buf)

function Base.setindex!(gpu::CuRefValue{T}, x::T) where {T}
    cpu = Ref(x)
    GC.@preserve cpu begin
        cpu_ptr = Base.unsafe_convert(Ptr{T}, cpu)
        gpu_ptr = Base.unsafe_convert(CuPtr{T}, gpu)
        unsafe_copyto!(gpu_ptr, cpu_ptr, 1; async=true)
    end
    return gpu
end

function Base.getindex(gpu::CuRefValue{T}) where {T}
    # synchronize first to maximize time spent executing Julia code
    synchronize(gpu.buf)

    cpu = Ref{T}()
    GC.@preserve cpu begin
        cpu_ptr = Base.unsafe_convert(Ptr{T}, cpu)
        gpu_ptr = Base.unsafe_convert(CuPtr{T}, gpu)
        unsafe_copyto!(cpu_ptr, gpu_ptr, 1; async=false)
    end
    cpu[]
end

function Base.show(io::IO, x::CuRefValue{T}) where {T}
    print(io, "CuRefValue{$T}(")
    print(io, x[])
    print(io, ")")
end


## reference backed by a CUDA array at index i

struct CuRefArray{T,A<:AbstractArray{T}} <: AbstractCuRef{T}
    x::A
    i::Int
    CuRefArray{T,A}(x,i) where {T,A<:AbstractArray{T}} = new(x,i)
end
CuRefArray{T}(x::AbstractArray{T}, i::Int=1) where {T} = CuRefArray{T,typeof(x)}(x, i)
CuRefArray(x::AbstractArray{T}, i::Int=1) where {T} = CuRefArray{T}(x, i)

Base.convert(::Type{CuRef{T}}, x::AbstractArray{T}) where {T} = CuRefArray(x, 1)
Base.convert(::Type{CuRef{T}}, x::CuRefArray{T}) where {T} = x

Base.unsafe_convert(P::Type{CuPtr{T}}, b::CuRefArray{T}) where {T} = pointer(b.x, b.i)
Base.unsafe_convert(P::Type{CuPtr{Any}}, b::CuRefArray{Any}) = convert(P, pointer(b.x, b.i))
Base.unsafe_convert(::Type{CuPtr{Cvoid}}, b::CuRefArray{T}) where {T} =
    convert(CuPtr{Cvoid}, Base.unsafe_convert(CuPtr{T}, b))

function Base.setindex!(gpu::CuRefArray{T}, x::T) where {T}
    cpu = Ref(x)
    GC.@preserve cpu begin
        cpu_ptr = Base.unsafe_convert(Ptr{T}, cpu)
        gpu_ptr = pointer(gpu.x, gpu.i)
        unsafe_copyto!(gpu_ptr, cpu_ptr, 1; async=true)
    end
    return gpu
end

function Base.getindex(gpu::CuRefArray{T}) where {T}
    # synchronize first to maximize time spent executing Julia code
    synchronize(gpu.x)

    cpu = Ref{T}()
    GC.@preserve cpu begin
        cpu_ptr = Base.unsafe_convert(Ptr{T}, cpu)
        gpu_ptr = pointer(gpu.x, gpu.i)
        unsafe_copyto!(cpu_ptr, gpu_ptr, 1; async=false)
    end
    cpu[]
end

function Base.show(io::IO, x::CuRefArray{T}) where {T}
    print(io, "CuRefArray{$T}(")
    print(io, x[])
    print(io, ")")
end
