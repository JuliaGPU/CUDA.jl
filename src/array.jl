import CUDAnative: DevicePtr

mutable struct CuArray{T,N} <: GPUArray{T,N}
  buf::Mem.Buffer
  own::Bool

  dims::Dims{N}
  offset::Int

  function CuArray{T,N}(buf::Mem.Buffer, dims::Dims{N}; offset::Integer=0, own::Bool=true) where {T,N}
    xs = new{T,N}(buf, own, dims, offset)
    if own
      Mem.retain(buf)
      finalizer(unsafe_free!, xs)
    end
    return xs
  end
end

CuVector{T} = CuArray{T,1}
CuMatrix{T} = CuArray{T,2}
CuVecOrMat{T} = Union{CuVector{T},CuMatrix{T}}

const INVALID = Mem.alloc(0)

function unsafe_free!(xs::CuArray{<:Any,N}) where {N}
  xs.buf === INVALID && return
  Mem.release(xs.buf) && dealloc(xs.buf, prod(xs.dims)*sizeof(eltype(xs)))
  xs.dims = Tuple(0 for _ in 1:N)
  xs.buf = INVALID
  return
end


## construction

# type and dimensionality specified, accepting dims as tuples of Ints
CuArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N} =
  CuArray{T,N}(alloc(prod(dims)*sizeof(T)), dims)

# type and dimensionality specified, accepting dims as series of Ints
CuArray{T,N}(::UndefInitializer, dims::Integer...) where {T,N} = CuArray{T,N}(undef, dims)

# type but not dimensionality specified
CuArray{T}(::UndefInitializer, dims::Dims{N}) where {T,N} = CuArray{T,N}(undef, dims)
CuArray{T}(::UndefInitializer, dims::Integer...) where {T} =
  CuArray{T}(undef, convert(Tuple{Vararg{Int}}, dims))

# empty vector constructor
CuArray{T,1}() where {T} = CuArray{T,1}(undef, 0)

# do-block constructors
for (ctor, tvars) in (:CuArray => (), :(CuArray{T}) => (:T,), :(CuArray{T,N}) => (:T, :N))
  @eval begin
    function $ctor(f::Function, args...) where {$(tvars...)}
      xs = $ctor(args...)
      try
        f(xs)
      finally
        unsafe_free!(xs)
      end
    end
  end
end


Base.similar(a::CuArray{T,N}) where {T,N} = CuArray{T,N}(undef, size(a))
Base.similar(a::CuArray{T}, dims::Base.Dims{N}) where {T,N} = CuArray{T,N}(undef, dims)
Base.similar(a::CuArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} = CuArray{T,N}(undef, dims)


"""
  unsafe_wrap(::CuArray, ptr::CuPtr{T}, dims; own=false, ctx=CuCurrentContext())

Wrap a `CuArray` object around the data at the address given by `ptr`. The pointer
element type `T` determines the array element type. `dims` is either an integer (for a 1d
array) or a tuple of the array dimensions. `own` optionally specified whether Julia should
take ownership of the memory, calling `free` when the array is no longer referenced. The
`ctx` argument determines the CUDA context where the data is allocated in.
"""
function Base.unsafe_wrap(::Union{Type{CuArray},Type{CuArray{T}},Type{CuArray{T,N}}},
                          p::CuPtr{T}, dims::NTuple{N,Int};
                          own::Bool=false, ctx::CuContext=CuCurrentContext()) where {T,N}
  buf = Mem.Buffer(convert(CuPtr{Cvoid}, p), prod(dims) * sizeof(T), ctx)
  return CuArray{T, length(dims)}(buf, dims; own=own)
end
function Base.unsafe_wrap(Atype::Union{Type{CuArray},Type{CuArray{T}},Type{CuArray{T,1}}},
                          p::CuPtr{T}, dim::Integer;
                          own::Bool=false, ctx::CuContext=CuCurrentContext()) where {T}
  unsafe_wrap(Atype, p, (dim,); own=own, ctx=ctx)
end
Base.unsafe_wrap(T::Type{<:CuArray}, ::Ptr, dims::NTuple{N,Int}; kwargs...) where {N} =
  throw(ArgumentError("cannot wrap a CPU pointer with a $T"))


## array interface

Base.elsize(::Type{<:CuArray{T}}) where {T} = sizeof(T)

Base.size(x::CuArray) = x.dims
Base.sizeof(x::CuArray) = Base.elsize(x) * length(x)


## interop with other arrays

CuArray{T,N}(xs::AbstractArray{T,N}) where {T,N} =
  isbits(xs) ?
    (CuArray{T,N}(undef, size(xs)) .= xs) :
    copyto!(CuArray{T,N}(undef, size(xs)), collect(xs))

CuArray{T,N}(xs::AbstractArray{S,N}) where {T,N,S} = CuArray{T,N}((x -> T(x)).(xs))

# underspecified constructors
CuArray{T}(xs::AbstractArray{S,N}) where {T,N,S} = CuArray{T,N}(xs)
(::Type{CuArray{T,N} where T})(x::AbstractArray{S,N}) where {S,N} = CuArray{S,N}(x)
CuArray(A::AbstractArray{T,N}) where {T,N} = CuArray{T,N}(A)

# idempotency
CuArray{T,N}(xs::CuArray{T,N}) where {T,N} = xs


## conversions

Base.convert(::Type{T}, x::T) where T <: CuArray = x

function Base._reshape(parent::CuArray, dims::Dims)
  n = length(parent)
  prod(dims) == n || throw(DimensionMismatch("parent has $n elements, which is incompatible with size $dims"))
  return CuArray{eltype(parent),length(dims)}(parent.buf, dims;
                                              offset=parent.offset, own=parent.own)
end
function Base._reshape(parent::CuArray{T,1}, dims::Tuple{Int}) where T
  n = length(parent)
  prod(dims) == n || throw(DimensionMismatch("parent has $n elements, which is incompatible with size $dims"))
  return parent
end


## interop with C libraries

"""
  buffer(array::CuArray [, index])

Get the native address of a CuArray, optionally at a given location `index`.
Equivalent of `Base.pointer` on `Array`s.
"""
function buffer(xs::CuArray, index=1)
  extra_offset = (index-1) * Base.elsize(xs)
  Mem.Buffer(xs.buf.ptr + xs.offset + extra_offset,
             sizeof(xs) - extra_offset,
             xs.buf.ctx)
end

Base.cconvert(::Type{<:Ptr}, x::CuArray) = throw(ArgumentError("cannot take the CPU address of a $(typeof(x))"))
Base.cconvert(::Type{<:CuPtr}, x::CuArray) = buffer(x)


## interop with CUDAnative

function Base.convert(::Type{CuDeviceArray{T,N,AS.Global}}, a::CuArray{T,N}) where {T,N}
  ptr = Base.unsafe_convert(CuPtr{T}, buffer(a))
  CuDeviceArray{T,N,AS.Global}(a.dims, DevicePtr{T,AS.Global}(ptr))
end

Adapt.adapt_storage(::CUDAnative.Adaptor, xs::CuArray{T,N}) where {T,N} =
  convert(CuDeviceArray{T,N,AS.Global}, xs)



## interop with CPU array

# We don't convert isbits types in `adapt`, since they are already
# considered GPU-compatible.

Adapt.adapt_storage(::Type{<:CuArray}, xs::AbstractArray) =
  isbits(xs) ? xs : convert(CuArray, xs)

Adapt.adapt_storage(::Type{<:CuArray{T}}, xs::AbstractArray{<:Real}) where T <: AbstractFloat =
  isbits(xs) ? xs : convert(CuArray{T}, xs)

Adapt.adapt_storage(::Type{<:Array}, xs::CuArray) = convert(Array, xs)

Base.collect(x::CuArray{T,N}) where {T,N} = copyto!(Array{T,N}(undef, size(x)), x)

function Base.unsafe_copyto!(dest::CuArray{T}, doffs, src::Array{T}, soffs, n) where T
  Mem.upload!(buffer(dest, doffs), pointer(src, soffs), n*sizeof(T))
  return dest
end

function Base.unsafe_copyto!(dest::Array{T}, doffs, src::CuArray{T}, soffs, n) where T
  Mem.download!(pointer(dest, doffs), buffer(src, soffs), n*sizeof(T))
  return dest
end

function Base.unsafe_copyto!(dest::CuArray{T}, doffs, src::CuArray{T}, soffs, n) where T
  Mem.transfer!(buffer(dest, doffs), buffer(src, soffs), n*sizeof(T))
  return dest
end

function Base.deepcopy_internal(x::CuArray, dict::IdDict)
  haskey(dict, x) && return dict[x]::typeof(x)
  return dict[x] = copy(x)
end


## utilities

cu(xs) = adapt(CuArray{Float32}, xs)
Base.getindex(::typeof(cu), xs...) = CuArray([xs...])

cuzeros(T::Type, dims...) = fill!(CuArray{T}(undef, dims...), 0)
cuones(T::Type, dims...) = fill!(CuArray{T}(undef, dims...), 1)
cuzeros(dims...) = cuzeros(Float32, dims...)
cuones(dims...) = cuones(Float32, dims...)
cufill(v, dims...) = fill!(CuArray{typeof(v)}(undef, dims...), v)
cufill(v, dims::Dims) = fill!(CuArray{typeof(v)}(undef, dims...), v)

# optimized implementation of `fill!` for types that are directly supported by memset
const MemsetTypes = Dict(1=>UInt8, 2=>UInt16, 4=>UInt32)
const MemsetCompatTypes = Union{UInt8, Int8,
                                UInt16, Int16, Float16,
                                UInt32, Int32, Float32}
function Base.fill!(A::CuArray{T}, x) where T <: MemsetCompatTypes
  y = reinterpret(MemsetTypes[sizeof(T)], convert(T, x))
  Mem.set!(buffer(A), y, length(A))
  A
end


## generic linear algebra routines

function LinearAlgebra.tril!(A::CuMatrix{T}, d::Integer = 0) where T
  function kernel!(_A, _d)
    li = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    m, n = size(_A)
    if 0 < li <= m*n
      i, j = Tuple(CartesianIndices(_A)[li])
      if i < j - _d
        _A[i, j] = 0
      end
    end
    return nothing
  end

  blk, thr = cudims(A)
  @cuda blocks=blk threads=thr kernel!(A, d)
  return A
end

function LinearAlgebra.triu!(A::CuMatrix{T}, d::Integer = 0) where T
  function kernel!(_A, _d)
    li = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    m, n = size(_A)
    if 0 < li <= m*n
      i, j = Tuple(CartesianIndices(_A)[li])
      if j < i + _d
        _A[i, j] = 0
      end
    end
    return nothing
  end

  blk, thr = cudims(A)
  @cuda blocks=blk threads=thr kernel!(A, d)
  return A
end
