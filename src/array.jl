export CuArray, CuVector, CuMatrix, CuVecOrMat, cu

mutable struct CuArray{T,N} <: AbstractGPUArray{T,N}
  ptr::CuPtr{T}
  dims::Dims{N}

  parent::Union{Nothing, CuArray} # parent array, for memory ownership tracking
  pooled::Bool                    # is this memory backed by the memory pool?

  # for early freeing outside of the GC
  refcount::Int
  freed::Bool

  ctx::CuContext

  # primary array
  function CuArray{T,N}(ptr::CuPtr{T}, dims::Dims{N}, pooled::Bool=true;
                        ctx=context()) where {T,N}
    self = new(ptr, dims, nothing, pooled, 0, false, ctx)
    retain(self)
    finalizer(unsafe_free!, self)
    return self
  end

  # derived array (e.g. view, reinterpret, ...)
  function CuArray{T,N}(ptr::CuPtr{T}, dims::Dims{N}, parent::CuArray) where {T,N}
    self = new(ptr, dims, parent, parent.pooled, 0, false, parent.ctx)
    retain(self)
    retain(parent)
    finalizer(unsafe_free!, self)
    return self
  end
end

function unsafe_free!(xs::CuArray)
  # this call should only have an effect once, becuase both the user and the GC can call it
  xs.freed && return
  _unsafe_free!(xs)
  xs.freed = true
  return
end

function _unsafe_free!(xs::CuArray)
  @assert xs.refcount >= 0
  if release(xs)
    if xs.parent === nothing
      # primary array with all references gone
      if xs.pooled && isvalid(xs.ctx)
        free(convert(CuPtr{Nothing}, pointer(xs)))
      end
    else
      # derived object
      _unsafe_free!(xs.parent)
    end

    # the object is dead, so we can also wipe the pointer
    xs.ptr = CU_NULL
  end

  return
end

@inline function retain(a::CuArray)
  a.refcount += 1
  return
end

@inline function release(a::CuArray)
  a.refcount -= 1
  return a.refcount == 0
end

Base.parent(A::CuArray) where {P} = something(A.parent, A)

function Base.dataids(A::CuArray)
  if A.parent === nothing
    (UInt(pointer(A)),)
  else
    (Base.dataids(parent(A))..., UInt(pointer(A)),)
  end
end

function Base.unaliascopy(A::CuArray) where {P}
  if A.parent === nothing
    copy(A)
  else
    offset = pointer(A) - pointer(A.parent)
    new_parent = Base.unaliascopy(A.parent)
    typeof(A)(pointer(new_parent) + offset, A.dims, new_parent)
  end
end

# optimized alias detection for views
function Base.mightalias(A::CuArray, B::CuArray)
    if parent(A) !== parent(B)
        # We cannot do any better than the usual dataids check
        return invoke(Base.mightalias, Tuple{AbstractArray, AbstractArray}, A, B)
    end

    rA = pointer(A):pointer(A)+sizeof(A)
    rB = pointer(B):pointer(B)+sizeof(B)
    return first(rA) <= first(rB) < last(rA) || first(rB) <= first(rA) < last(rB)
end


## convenience constructors

CuVector{T} = CuArray{T,1}
CuMatrix{T} = CuArray{T,2}
CuVecOrMat{T} = Union{CuVector{T},CuMatrix{T}}

# type and dimensionality specified, accepting dims as tuples of Ints
function CuArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N}
  Base.isbitsunion(T) && error("CuArray does not yet support union bits types")
  Base.isbitstype(T)  || error("CuArray only supports bits types") # allocatedinline on 1.3+
  ptr = alloc(prod(dims) * sizeof(T))
  CuArray{T,N}(convert(CuPtr{T}, ptr), dims)
end

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

function Base.copy(a::CuArray{T,N}) where {T,N}
  ptr = convert(CuPtr{T}, alloc(sizeof(a)))
  unsafe_copyto!(ptr, pointer(a), length(a); async=true, stream=CuStreamPerThread())
  CuArray{T,N}(ptr, size(a))
end


"""
  unsafe_wrap(::CuArray, ptr::CuPtr{T}, dims; own=false, ctx=context())

Wrap a `CuArray` object around the data at the address given by `ptr`. The pointer
element type `T` determines the array element type. `dims` is either an integer (for a 1d
array) or a tuple of the array dimensions. `own` optionally specified whether Julia should
take ownership of the memory, calling `cudaFree` when the array is no longer referenced. The
`ctx` argument determines the CUDA context where the data is allocated in.
"""
function Base.unsafe_wrap(::Union{Type{CuArray},Type{CuArray{T}},Type{CuArray{T,N}}},
                          p::CuPtr{T}, dims::NTuple{N,Int};
                          own::Bool=false, ctx::CuContext=context()) where {T,N}
  xs = CuArray{T, length(dims)}(p, dims, false; ctx=ctx)
  if own
    base = convert(CuPtr{Cvoid}, p)
    buf = Mem.DeviceBuffer(base, prod(dims) * sizeof(T))
    finalizer(xs) do obj
      if isvalid(obj.ctx)
        Mem.free(buf)
      end
    end
  end
  return xs
end

function Base.unsafe_wrap(Atype::Union{Type{CuArray},Type{CuArray{T}},Type{CuArray{T,1}}},
                          p::CuPtr{T}, dim::Integer;
                          own::Bool=false, ctx::CuContext=context()) where {T}
  unsafe_wrap(Atype, p, (dim,); own=own, ctx=ctx)
end

Base.unsafe_wrap(T::Type{<:CuArray}, ::Ptr, dims::NTuple{N,Int}; kwargs...) where {N} =
  throw(ArgumentError("cannot wrap a CPU pointer with a $T"))


## array interface

Base.elsize(::Type{<:CuArray{T}}) where {T} = sizeof(T)

Base.size(x::CuArray) = x.dims
Base.sizeof(x::CuArray) = Base.elsize(x) * length(x)


## derived types

export DenseCuArray, DenseCuVector, DenseCuMatrix, DenseCuVecOrMat,
       StridedCuArray, StridedCuVector, StridedCuMatrix, StridedCuVecOrMat,
       WrappedCuArray, WrappedCuVector, WrappedCuMatrix, WrappedCuVecOrMat

ContiguousSubCuArray{T,N,A<:CuArray} = Base.FastContiguousSubArray{T,N,A}

# dense arrays: stored contiguously in memory
DenseReinterpretCuArray{T,N,A<:Union{CuArray,ContiguousSubCuArray}} = Base.ReinterpretArray{T,N,S,A} where S
DenseReshapedCuArray{T,N,A<:Union{CuArray,ContiguousSubCuArray,DenseReinterpretCuArray}} = Base.ReshapedArray{T,N,A}
DenseSubCuArray{T,N,A<:Union{CuArray,DenseReshapedCuArray,DenseReinterpretCuArray}} = Base.FastContiguousSubArray{T,N,A}
DenseCuArray{T,N} = Union{CuArray{T,N}, DenseSubCuArray{T,N}, DenseReshapedCuArray{T,N}, DenseReinterpretCuArray{T,N}}
DenseCuVector{T} = DenseCuArray{T,1}
DenseCuMatrix{T} = DenseCuArray{T,2}
DenseCuVecOrMat{T} = Union{DenseCuVector{T}, DenseCuMatrix{T}}

# strided arrays
StridedSubCuArray{T,N,A<:Union{CuArray,DenseReshapedCuArray,DenseReinterpretCuArray},
                  I<:Tuple{Vararg{Union{Base.RangeIndex, Base.ReshapedUnitRange,
                                        Base.AbstractCartesianIndex}}}} = SubArray{T,N,A,I}
StridedCuArray{T,N} = Union{CuArray{T,N}, StridedSubCuArray{T,N}, DenseReshapedCuArray{T,N}, DenseReinterpretCuArray{T,N}}
StridedCuVector{T} = StridedCuArray{T,1}
StridedCuMatrix{T} = StridedCuArray{T,2}
StridedCuVecOrMat{T} = Union{StridedCuVector{T}, StridedCuMatrix{T}}

Base.pointer(x::StridedCuArray{T}) where {T} = Base.unsafe_convert(CuPtr{T}, x)
@inline function Base.pointer(x::StridedCuArray{T}, i::Integer) where T
    Base.unsafe_convert(CuPtr{T}, x) + Base._memory_offset(x, i)
end

# wrapped arrays: can be used in kernels
WrappedCuArray{T,N} = Union{CuArray{T,N}, WrappedArray{T,N,CuArray,CuArray{T,N}}}
WrappedCuVector{T} = WrappedCuArray{T,1}
WrappedCuMatrix{T} = WrappedCuArray{T,2}
WrappedCuVecOrMat{T} = Union{WrappedCuVector{T}, WrappedCuMatrix{T}}


## interop with other arrays

@inline function CuArray{T,N}(xs::AbstractArray{<:Any,N}) where {T,N}
  A = CuArray{T,N}(undef, size(xs))
  copyto!(A, convert(Array{T}, xs))
  return A
end

# underspecified constructors
CuArray{T}(xs::AbstractArray{S,N}) where {T,N,S} = CuArray{T,N}(xs)
(::Type{CuArray{T,N} where T})(x::AbstractArray{S,N}) where {S,N} = CuArray{S,N}(x)
CuArray(A::AbstractArray{T,N}) where {T,N} = CuArray{T,N}(A)

# idempotency
CuArray{T,N}(xs::CuArray{T,N}) where {T,N} = xs


## conversions

Base.convert(::Type{T}, x::T) where T <: CuArray = x


## interop with C libraries

Base.unsafe_convert(::Type{Ptr{T}}, x::CuArray{T}) where {T} = throw(ArgumentError("cannot take the CPU address of a $(typeof(x))"))
Base.unsafe_convert(::Type{CuPtr{T}}, x::CuArray{T}) where {T} = x.ptr


## interop with device arrays

function Base.unsafe_convert(::Type{CuDeviceArray{T,N,AS.Global}}, a::DenseCuArray{T,N}) where {T,N}
  CuDeviceArray{T,N,AS.Global}(size(a), reinterpret(LLVMPtr{T,AS.Global}, pointer(a)))
end

Adapt.adapt_storage(::Adaptor, xs::CuArray{T,N}) where {T,N} =
  Base.unsafe_convert(CuDeviceArray{T,N,AS.Global}, xs)

# we materialize ReshapedArray/ReinterpretArray/SubArray/... directly as a device array
Adapt.adapt_structure(::Adaptor, xs::DenseCuArray{T,N}) where {T,N} =
  Base.unsafe_convert(CuDeviceArray{T,N,AS.Global}, xs)


## interop with CPU arrays

# We don't convert isbits types in `adapt`, since they are already
# considered GPU-compatible.

Adapt.adapt_storage(::Type{CuArray}, xs::AT) where {AT<:AbstractArray} =
  isbitstype(AT) ? xs : convert(CuArray, xs)

# if an element type is specified, convert to it
Adapt.adapt_storage(::Type{<:CuArray{T}}, xs::AT) where {T, AT<:AbstractArray} =
  isbitstype(AT) ? xs : convert(CuArray{T}, xs)

Adapt.adapt_storage(::Type{Array}, xs::CuArray) = convert(Array, xs)

Base.collect(x::CuArray{T,N}) where {T,N} = copyto!(Array{T,N}(undef, size(x)), x)

function Base.copyto!(dest::DenseCuArray{T}, doffs::Integer, src::Array{T}, soffs::Integer,
                      n::Integer) where T
  n==0 && return dest
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  unsafe_copyto!(dest, doffs, src, soffs, n)
  return dest
end

Base.copyto!(dest::DenseCuArray{T}, src::Array{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.copyto!(dest::Array{T}, doffs::Integer, src::DenseCuArray{T}, soffs::Integer,
                      n::Integer) where T
  n==0 && return dest
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  unsafe_copyto!(dest, doffs, src, soffs, n)
  return dest
end

Base.copyto!(dest::Array{T}, src::DenseCuArray{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.copyto!(dest::DenseCuArray{T}, doffs::Integer, src::DenseCuArray{T}, soffs::Integer,
                      n::Integer) where T
  n==0 && return dest
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  unsafe_copyto!(dest, doffs, src, soffs, n)
  return dest
end

Base.copyto!(dest::DenseCuArray{T}, src::DenseCuArray{T}) where {T} =
    copyto!(dest, 1, src, 1, length(src))

function Base.unsafe_copyto!(dest::DenseCuArray{T}, doffs, src::Array{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("Not implemented")
  end
  return dest
end

function Base.unsafe_copyto!(dest::Array{T}, doffs, src::DenseCuArray{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("Not implemented")
  end
  return dest
end

function Base.unsafe_copyto!(dest::DenseCuArray{T}, doffs, src::DenseCuArray{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n;
                                       async=true, stream=CuStreamPerThread())
  if Base.isbitsunion(T)
    # copy selector bytes
    error("Not implemented")
  end
  return dest
end

function Base.deepcopy_internal(x::CuArray, dict::IdDict)
  haskey(dict, x) && return dict[x]::typeof(x)
  return dict[x] = copy(x)
end


## Float32-preferring conversion

struct Float32Adaptor end

Adapt.adapt_storage(::Float32Adaptor, xs::AbstractArray) =
  isbits(xs) ? xs : convert(CuArray, xs)

Adapt.adapt_storage(::Float32Adaptor, xs::AbstractArray{<:AbstractFloat}) =
  isbits(xs) ? xs : convert(CuArray{Float32}, xs)

# not for Float16
Adapt.adapt_storage(::Float32Adaptor, xs::AbstractArray{Float16}) =
  isbits(xs) ? xs : convert(CuArray, xs)
Adapt.adapt_storage(::Float32Adaptor, xs::AbstractArray{BFloat16}) =
  isbits(xs) ? xs : convert(CuArray, xs)

cu(xs) = adapt(Float32Adaptor(), xs)
Base.getindex(::typeof(cu), xs...) = CuArray([xs...])


## utilities

zeros(T::Type, dims...) = fill!(CuArray{T}(undef, dims...), 0)
ones(T::Type, dims...) = fill!(CuArray{T}(undef, dims...), 1)
zeros(dims...) = zeros(Float32, dims...)
ones(dims...) = ones(Float32, dims...)
fill(v, dims...) = fill!(CuArray{typeof(v)}(undef, dims...), v)
fill(v, dims::Dims) = fill!(CuArray{typeof(v)}(undef, dims...), v)

# optimized implementation of `fill!` for types that are directly supported by memset
memsettype(T::Type) = T
memsettype(T::Type{<:Signed}) = unsigned(T)
memsettype(T::Type{<:AbstractFloat}) = Base.uinttype(T)
const MemsetCompatTypes = Union{UInt8, Int8,
                                UInt16, Int16, Float16,
                                UInt32, Int32, Float32}
function Base.fill!(A::DenseCuArray{T}, x) where T <: MemsetCompatTypes
  U = memsettype(T)
  y = reinterpret(U, convert(T, x))
  Mem.set!(convert(CuPtr{U}, pointer(A)), y, length(A))
  A
end


## views

@inline function Base.view(A::CuArray, I::Vararg{Any,N}) where {N}
    J = to_indices(A, I)
    @boundscheck begin
        # Base's boundscheck accesses the indices, so make sure they reside on the CPU.
        # this is expensive, but it's a bounds check after all.
        J_cpu = map(j->adapt(Array, j), J)
        checkbounds(A, J_cpu...)
    end
    J_gpu = map(j->adapt(CuArray, j), J)
    Base.unsafe_view(Base._maybe_reshape_parent(A, Base.index_ndims(J_gpu...)), J_gpu...)
end

# upload the SubArray indices when adapting to the GPU
# (can't do this eagerly or the view constructor wouldn't be able to boundscheck)
Adapt.adapt_structure(to::Adaptor, A::SubArray) =
    SubArray(adapt(to, parent(A)), adapt(to, adapt(CuArray, parentindices(A))))

function Base.unsafe_convert(::Type{CuPtr{T}}, V::SubArray{T,N,P,<:Tuple{Vararg{Base.RangeIndex}}}) where {T,N,P<:CuArray}
    return Base.unsafe_convert(CuPtr{T}, parent(V)) +
           Base._memory_offset(V.parent, map(first, V.indices)...)
end


## reshape

Base.unsafe_convert(::Type{CuPtr{T}}, a::Base.ReshapedArray{T}) where {T} =
  Base.unsafe_convert(CuPtr{T}, parent(a))


## reinterpret

Base.unsafe_convert(::Type{CuPtr{T}}, a::Base.ReinterpretArray{T,N,S} where N) where {T,S} =
  CuPtr{T}(Base.unsafe_convert(CuPtr{S}, parent(a)))


## resizing

"""
  resize!(a::CuVector, n::Int)

Resize `a` to contain `n` elements. If `n` is smaller than the current collection length,
the first `n` elements will be retained. If `n` is larger, the new elements are not
guaranteed to be initialized.

Several restrictions apply to which types of `CuArray`s can be resized:

- the array should be backed by the memory pool, and not have been constructed with `unsafe_wrap`
- the array cannot be derived (view, reshape) from another array
- the array cannot have any derived arrays itself

"""
function Base.resize!(A::CuVector{T}, n::Int) where T
  A.parent === nothing || error("cannot resize derived CuArray")
  A.refcount == 1 || error("cannot resize shared CuArray")
  A.pooled || error("cannot resize wrapped CuArray")

  ptr = convert(CuPtr{T}, alloc(n * sizeof(T)))
  m = Base.min(length(A), n)
  unsafe_copyto!(ptr, pointer(A), m)

  free(convert(CuPtr{Nothing}, pointer(A)))
  A.dims = (n,)
  A.ptr = ptr

  A
end
