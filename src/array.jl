export CuArray, CuVector, CuMatrix, CuVecOrMat, cu, is_unified


## array storage

# array storage is shared by arrays that refer to the same data, while keeping track of
# the number of outstanding references

struct ArrayStorage{B}
  buffer::B

  ctx::CuContext

  # the refcount also encodes the state of the array:
  # < 0: unmanaged
  # = 0: freed
  # > 0: referenced
  refcount::Threads.Atomic{Int}
end

ArrayStorage(buf::B, ctx, state::Int) where {B} =
  ArrayStorage{B}(buf, ctx, Threads.Atomic{Int}(state))


## array type

mutable struct CuArray{T,N,B} <: AbstractGPUArray{T,N}
  storage::Union{Nothing,ArrayStorage{B}}

  maxsize::Int  # maximum data size; excluding any selector bytes
  offset::Int

  dims::Dims{N}

  function CuArray{T,N,B}(::UndefInitializer, dims::Dims{N}) where {T,N,B}
    Base.allocatedinline(T) || error("CuArray only supports element types that are stored inline")
    maxsize = prod(dims) * sizeof(T)
    bufsize = if Base.isbitsunion(T)
      # type tag array past the data
      maxsize + prod(dims)
    else
      maxsize
    end
    buf = alloc(B, bufsize)
    storage = ArrayStorage(buf, context(), 1)
    obj = new{T,N,B}(storage, maxsize, 0, dims)
    finalizer(unsafe_finalize!, obj)
  end

  function CuArray{T,N}(storage::ArrayStorage{B}, dims::Dims{N};
                        maxsize::Int=prod(dims) * sizeof(T), offset::Int=0) where {T,N,B}
    Base.allocatedinline(T) || error("CuArray only supports element types that are stored inline")
    return new{T,N,B}(storage, maxsize, offset, dims)
  end
end

"""
    CUDA.unsafe_free!(a::CuArray, [stream::CuStream])

Release the memory of an array for reuse by future allocations. This function is
automatically called by the finalizer when an array goes out of scope, but can be called
earlier to reduce pressure on the memory allocator.

By default, the operation is performed on the task-local stream. During task or process
finalization however, that stream may be destroyed already, so be sure to specify a safe
stream (i.e. `default_stream()`, which will ensure the operation will block on other
streams) when calling this function from a finalizer. For simplicity, the `unsafe_finalize!`
function does exactly that.
"""
function unsafe_free!(xs::CuArray, stream::CuStream=stream())
  # this call should only have an effect once, because both the user and the GC can call it
  if xs.storage === nothing
    return
  elseif xs.storage.refcount[] < 0
    throw(ArgumentError("Cannot free an unmanaged buffer."))
  end

  refcount = Threads.atomic_add!(xs.storage.refcount, -1)
  if refcount == 1
    @context! skip_destroyed=true xs.storage.ctx begin
      free(xs.storage.buffer; stream)
    end
  end

  # this array object is now dead, so replace its storage by a dummy one
  xs.storage = nothing

  return
end

function unsafe_finalize!(xs::CuArray)
  # during task or process finalization, the local stream might be destroyed already, so
  # use the default stream. additionally, since we don't use per-thread APIs, this default
  # stream follows legacy semantics and will synchronize all other streams. this protects
  # against freeing resources that are still in use.
  #
  # TODO: although this is still an asynchronous operation, even when using the default
  # stream, it synchronizes "too much". we could do better, e.g., by keeping track of all
  # streams involved, or by refcounting uses and decrementing that refcount after the
  # operation using `cuLaunchHostFunc`. See CUDA.jl#778 and CUDA.jl#780 for details.
  unsafe_free!(xs, default_stream())
end


## alias detection

Base.dataids(A::CuArray) = (UInt(pointer(A)),)

Base.unaliascopy(A::CuArray) = copy(A)

function Base.mightalias(A::CuArray, B::CuArray)
  rA = pointer(A):pointer(A)+sizeof(A)
  rB = pointer(B):pointer(B)+sizeof(B)
  return first(rA) <= first(rB) < last(rA) || first(rB) <= first(rA) < last(rB)
end


## convenience constructors

CuVector{T} = CuArray{T,1}
CuMatrix{T} = CuArray{T,2}
CuVecOrMat{T} = Union{CuVector{T},CuMatrix{T}}

# default to non-unified memory
CuArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N} =
  CuArray{T,N,Mem.DeviceBuffer}(undef, dims)
is_unified(a::CuArray) = isa(a.storage.buffer, Mem.UnifiedBuffer)

# type and dimensionality specified, accepting dims as series of Ints
CuArray{T,N,B}(::UndefInitializer, dims::Integer...) where {T,N,B} =
  CuArray{T,N,B}(undef, dims)
CuArray{T,N}(::UndefInitializer, dims::Integer...) where {T,N} =
  CuArray{T,N}(undef, dims)

# type but not dimensionality specified
CuArray{T}(::UndefInitializer, dims::Dims{N}) where {T,N} =
  CuArray{T,N}(undef, dims)
CuArray{T}(::UndefInitializer, dims::Integer...) where {T} =
  CuArray{T}(undef, convert(Tuple{Vararg{Int}}, dims))

# empty vector constructor
CuArray{T,1,B}() where {T,B} = CuArray{T,1,B}(undef, 0)
CuArray{T,1}() where {T} = CuArray{T,1}(undef, 0)

# do-block constructors
for (ctor, tvars) in (:CuArray => (),
                      :(CuArray{T}) => (:T,),
                      :(CuArray{T,N}) => (:T, :N),
                      :(CuArray{T,N,B}) => (:T, :N, :B))
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

Base.similar(a::CuArray{T,N,B}) where {T,N,B} =
  CuArray{T,N,B}(undef, size(a))
Base.similar(a::CuArray{T,<:Any,B}, dims::Base.Dims{N}) where {T,N,B} =
  CuArray{T,N,B}(undef, dims)
Base.similar(a::CuArray{<:Any,<:Any,B}, ::Type{T}, dims::Base.Dims{N}) where {T,N,B} =
  CuArray{T,N,B}(undef, dims)

function Base.copy(a::CuArray{T,N}) where {T,N}
  b = similar(a)
  @inbounds copyto!(b, a)
end

# XXX: defining deepcopy_internal, as per the deepcopy documentation, results in a ton
#      of invalidations, so we redefine deepcopy itself (see JuliaGPU/CUDA.jl#632)
function Base.deepcopy(x::CuArray)
  dict = IdDict()
  haskey(dict, x) && return dict[x]::typeof(x)
  return dict[x] = copy(x)
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
                          ptr::CuPtr{T}, dims::NTuple{N,Int};
                          own::Bool=false, ctx::CuContext=context()) where {T,N}
  Base.isbitstype(T) || error("Can only unsafe_wrap a pointer to a bits type")
  sz = prod(dims) * sizeof(T)

  # identify the buffer
  buf = try
    typ = memory_type(ptr)
    if is_managed(ptr)
      Mem.UnifiedBuffer(ptr, sz)
    elseif typ == CU_MEMORYTYPE_DEVICE
      # TODO: can we identify whether this pointer was allocated asynchronously?
      Mem.DeviceBuffer(ptr, sz, false)
    elseif typ == CU_MEMORYTYPE_HOST
      Mem.HostBuffer(reinterpret(Ptr{T}, ptr), sz)
    else
      error("Unknown memory type; please file an issue.")
    end
  catch err
      error("Could not identify the buffer type; are you passing a valid CUDA pointer to unsafe_wrap?")
  end

  storage = ArrayStorage(buf, ctx, own ? 1 : -1)
  CuArray{T, length(dims)}(storage, dims)
end

function Base.unsafe_wrap(Atype::Union{Type{CuArray},Type{CuArray{T}},Type{CuArray{T,1}}},
                          p::CuPtr{T}, dim::Integer;
                          own::Bool=false, ctx::CuContext=context()) where {T}
  unsafe_wrap(Atype, p, (dim,); own, ctx)
end

Base.unsafe_wrap(T::Type{<:CuArray}, ::Ptr, dims::NTuple{N,Int}; kwargs...) where {N} =
  throw(ArgumentError("cannot wrap a CPU pointer with a $T"))


## array interface

Base.elsize(::Type{<:CuArray{T}}) where {T} = sizeof(T)

Base.size(x::CuArray) = x.dims
Base.sizeof(x::CuArray) = Base.elsize(x) * length(x)

function context(A::CuArray)
  A.storage === nothing && throw(UndefRefError())
  return A.storage.ctx
end

function device(A::CuArray)
  A.storage === nothing && throw(UndefRefError())
  return device(A.storage.ctx)
end


## derived types

export DenseCuArray, DenseCuVector, DenseCuMatrix, DenseCuVecOrMat,
       StridedCuArray, StridedCuVector, StridedCuMatrix, StridedCuVecOrMat,
       AnyCuArray, AnyCuVector, AnyCuMatrix, AnyCuVecOrMat

# dense arrays: stored contiguously in memory
#
# all common dense wrappers are currently represented as CuArray objects.
# this simplifies common use cases, and greatly improves load time.
# CUDA.jl 2.0 experimented with using ReshapedArray/ReinterpretArray/SubArray,
# but that proved much too costly. TODO: revisit when we have better Base support.
DenseCuArray{T,N} = CuArray{T,N}
DenseCuVector{T} = DenseCuArray{T,1}
DenseCuMatrix{T} = DenseCuArray{T,2}
DenseCuVecOrMat{T} = Union{DenseCuVector{T}, DenseCuMatrix{T}}
# XXX: these dummy aliases (DenseCuArray=CuArray) break alias printing, as
#      `Base.print_without_params` only handles the case of a single alias.

# strided arrays
StridedSubCuArray{T,N,I<:Tuple{Vararg{Union{Base.RangeIndex, Base.ReshapedUnitRange,
                                            Base.AbstractCartesianIndex}}}} =
  SubArray{T,N,<:CuArray,I}
StridedCuArray{T,N} = Union{CuArray{T,N}, StridedSubCuArray{T,N}}
StridedCuVector{T} = StridedCuArray{T,1}
StridedCuMatrix{T} = StridedCuArray{T,2}
StridedCuVecOrMat{T} = Union{StridedCuVector{T}, StridedCuMatrix{T}}

Base.pointer(x::StridedCuArray{T}) where {T} = Base.unsafe_convert(CuPtr{T}, x)
@inline function Base.pointer(x::StridedCuArray{T}, i::Integer) where T
    Base.unsafe_convert(CuPtr{T}, x) + Base._memory_offset(x, i)
end

# anything that's (secretly) backed by a CuArray
AnyCuArray{T,N} = Union{CuArray{T,N}, WrappedArray{T,N,CuArray,CuArray{T,N}}}
AnyCuVector{T} = AnyCuArray{T,1}
AnyCuMatrix{T} = AnyCuArray{T,2}
AnyCuVecOrMat{T} = Union{AnyCuVector{T}, AnyCuMatrix{T}}


## interop with other arrays

@inline function CuArray{T,N,B}(xs::AbstractArray{<:Any,N}) where {T,N,B}
  A = CuArray{T,N,B}(undef, size(xs))
  copyto!(A, convert(Array{T}, xs))
  return A
end

@inline CuArray{T,N}(xs::AbstractArray{<:Any,N}) where {T,N} =
  CuArray{T,N,Mem.Device}(xs)

@inline CuArray{T,N}(xs::CuArray{<:Any,N,B}) where {T,N,B} =
  CuArray{T,N,B}(xs)

# underspecified constructors
CuArray{T}(xs::AbstractArray{S,N}) where {T,N,S} = CuArray{T,N}(xs)
(::Type{CuArray{T,N} where T})(x::AbstractArray{S,N}) where {S,N} = CuArray{S,N}(x)
CuArray(A::AbstractArray{T,N}) where {T,N} = CuArray{T,N}(A)

# idempotency
CuArray{T,N,B}(xs::CuArray{T,N,B}) where {T,N,B} = xs
CuArray{T,N}(xs::CuArray{T,N,B}) where {T,N,B} = xs


## conversions

Base.convert(::Type{T}, x::T) where T <: CuArray = x


## interop with C libraries

Base.unsafe_convert(::Type{Ptr{T}}, x::CuArray{T}) where {T} =
  throw(ArgumentError("cannot take the CPU address of a $(typeof(x))"))
Base.unsafe_convert(::Type{CuPtr{T}}, x::CuArray{T}) where {T} =
  convert(CuPtr{T}, x.storage.buffer) + x.offset


## interop with device arrays

function Base.unsafe_convert(::Type{CuDeviceArray{T,N,AS.Global}}, a::DenseCuArray{T,N}) where {T,N}
  CuDeviceArray{T,N,AS.Global}(size(a), reinterpret(LLVMPtr{T,AS.Global}, pointer(a)),
                               a.maxsize - a.offset)
end


## memory copying

typetagdata(a::Array, i=1) = ccall(:jl_array_typetagdata, Ptr{UInt8}, (Any,), a) + i - 1
typetagdata(a::CuArray, i=1) =
  convert(CuPtr{UInt8}, a.storage.buffer) + a.maxsize + a.offset÷Base.elsize(a) + i - 1

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

# device memory

# NOTE: we only switch contexts here to avoid illegal memory accesses. synchronization is
#       best-effort, since we don't keep track of streams using each array.

function Base.unsafe_copyto!(dest::DenseCuArray{T,<:Any,Mem.DeviceBuffer}, doffs,
                             src::Array{T}, soffs, n) where T
  @context! context(dest) begin
    # operations on unpinned memory cannot be executed asynchronously, and synchronize
    # without yielding back to the Julia scheduler. prevent that by eagerly synchronizing.
    is_pinned(pointer(src)) || synchronize()

    GC.@preserve src dest begin
      unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n; async=true)
      if Base.isbitsunion(T)
        unsafe_copyto!(typetagdata(dest, doffs), typetagdata(src, soffs), n; async=true)
      end
    end
  end
  return dest
end

function Base.unsafe_copyto!(dest::Array{T}, doffs,
                             src::DenseCuArray{T,<:Any,Mem.DeviceBuffer}, soffs, n) where T
  @context! context(src) begin
    # operations on unpinned memory cannot be executed asynchronously, and synchronize
    # without yielding back to the Julia scheduler. prevent that by eagerly synchronizing.
    is_pinned(pointer(dest)) || synchronize()

    GC.@preserve src dest begin
      unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n; async=true)
      if Base.isbitsunion(T)
        unsafe_copyto!(typetagdata(dest, doffs), typetagdata(src, soffs), n; async=true)
      end
    end

    # users expect values to be available after this call
    synchronize()
  end
  return dest
end

function Base.unsafe_copyto!(dest::DenseCuArray{T,<:Any,Mem.DeviceBuffer}, doffs,
                             src::DenseCuArray{T,<:Any,Mem.DeviceBuffer}, soffs, n) where T
  context(dest) == context(src) || throw(ArgumentError("copying between arrays from different contexts"))
  @context! context(dest) begin
    GC.@preserve src dest begin
      unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n; async=true)
      if Base.isbitsunion(T)
        unsafe_copyto!(typetagdata(dest, doffs), typetagdata(src, soffs), n; async=true)
      end
    end
  end
  return dest
end

# unified memory

# NOTE: synchronization is best-effort, since we don't keep track of the
#       defices and streams using each array backed by unified memory.

function Base.unsafe_copyto!(dest::DenseCuArray{T,<:Any,Mem.UnifiedBuffer}, doffs,
                             src::Array{T}, soffs, n) where T
  # maintain stream-ordered semantics
  # XXX: alternative, use an async CUDA memcpy if the stream isn't idle?
  synchronize()

  GC.@preserve src dest begin
    cpu_ptr = pointer(src, soffs)
    unsafe_copyto!(reinterpret(typeof(cpu_ptr), pointer(dest, doffs)), cpu_ptr, n)
    if Base.isbitsunion(T)
      cpu_ptr = typetagdata(src, soffs)
      unsafe_copyto!(reinterpret(typeof(cpu_ptr), typetagdata(dest, doffs)), cpu_ptr, n)
    end
  end
  return dest
end

function Base.unsafe_copyto!(dest::Array{T}, doffs,
                             src::DenseCuArray{T,<:Any,Mem.UnifiedBuffer}, soffs, n) where T
  # maintain stream-ordered semantics
  synchronize()

  GC.@preserve src dest begin
    cpu_ptr = pointer(dest, doffs)
    unsafe_copyto!(cpu_ptr, reinterpret(typeof(cpu_ptr), pointer(src, soffs)), n)
    if Base.isbitsunion(T)
      cpu_ptr = typetagdata(dest, doffs)
      unsafe_copyto!(cpu_ptr, reinterpret(typeof(cpu_ptr), typetagdata(src, soffs)), n)
    end
  end

  return dest
end

# TODO: copying between CUDA arrays (unified<->unified, unified<->device, device<->unified)


## regular gpu array adaptor

# We don't convert isbits types in `adapt`, since they are already
# considered GPU-compatible.

Adapt.adapt_storage(::Type{CuArray}, xs::AT) where {AT<:AbstractArray} =
  isbitstype(AT) ? xs : convert(CuArray, xs)

# if an element type is specified, convert to it
Adapt.adapt_storage(::Type{<:CuArray{T}}, xs::AT) where {T, AT<:AbstractArray} =
  isbitstype(AT) ? xs : convert(CuArray{T}, xs)


## opinionated gpu array adaptor

# eagerly converts Float64 to Float32, for performance reasons

struct CuArrayAdaptor{B} end

Adapt.adapt_storage(::CuArrayAdaptor{B}, xs::AbstractArray{T,N}) where {T,N,B} =
  isbits(xs) ? xs : CuArray{T,N,B}(xs)

Adapt.adapt_storage(::CuArrayAdaptor{B}, xs::AbstractArray{T,N}) where {T<:AbstractFloat,N,B} =
  isbits(xs) ? xs : CuArray{Float32,N,B}(xs)

Adapt.adapt_storage(::CuArrayAdaptor{B}, xs::AbstractArray{T,N}) where {T<:Complex{<:AbstractFloat},N,B} =
  isbits(xs) ? xs : CuArray{ComplexF32,N,B}(xs)

# not for Float16
Adapt.adapt_storage(::CuArrayAdaptor{B}, xs::AbstractArray{T,N}) where {T<:Union{Float16,BFloat16},N,B} =
  isbits(xs) ? xs : CuArray{T,N,B}(xs)

@inline cu(xs; unified::Bool=false) = adapt(CuArrayAdaptor{unified ? Mem.UnifiedBuffer : Mem.DeviceBuffer}(), xs)
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
  @context! context(A) begin
    Mem.set!(convert(CuPtr{U}, pointer(A)), y, length(A))
  end
  A
end


## views

# optimize view to return a CuArray when contiguous

struct Contiguous end
struct NonContiguous end

# NOTE: this covers more cases than the I<:... in Base.FastContiguousSubArray
CuIndexStyle() = Contiguous()
CuIndexStyle(I...) = NonContiguous()
CuIndexStyle(::Union{Base.ScalarIndex, CartesianIndex}...) = Contiguous()
CuIndexStyle(i1::Colon, ::Union{Base.ScalarIndex, CartesianIndex}...) = Contiguous()
CuIndexStyle(i1::AbstractUnitRange, ::Union{Base.ScalarIndex, CartesianIndex}...) = Contiguous()
CuIndexStyle(i1::Colon, I...) = CuIndexStyle(I...)

cuviewlength() = ()
@inline cuviewlength(::Real, I...) = cuviewlength(I...) # skip scalar

if VERSION >= v"1.8.0-DEV.120"
@inline cuviewlength(i1::AbstractUnitRange, I...) = (Base.length(i1), cuviewlength(I...)...)
@inline cuviewlength(i1::AbstractUnitRange, ::Base.ScalarIndex...) = (Base.length(i1),)
else
@inline cuviewlength(i1::AbstractUnitRange, I...) = (length(i1), cuviewlength(I...)...)
@inline cuviewlength(i1::AbstractUnitRange, ::Base.ScalarIndex...) = (length(i1),)
end

# we don't really want an array, so don't call `adapt(Array, ...)`,
# but just want CuArray indices to get downloaded back to the CPU.
# this makes sure we preserve array-like containers, like Base.Slice.
struct BackToCPU end
Adapt.adapt_storage(::BackToCPU, xs::CuArray) = convert(Array, xs)

@inline function Base.view(A::CuArray, I::Vararg{Any,N}) where {N}
    J = to_indices(A, I)
    @boundscheck begin
        # Base's boundscheck accesses the indices, so make sure they reside on the CPU.
        # this is expensive, but it's a bounds check after all.
        J_cpu = map(j->adapt(BackToCPU(), j), J)
        checkbounds(A, J_cpu...)
    end
    J_gpu = map(j->adapt(CuArray, j), J)
    unsafe_view(A, J_gpu, CuIndexStyle(I...))
end

@inline function unsafe_view(A, I, ::Contiguous)
    unsafe_contiguous_view(Base._maybe_reshape_parent(A, Base.index_ndims(I...)), I, cuviewlength(I...))
end
@inline function unsafe_contiguous_view(a::CuArray{T}, I::NTuple{N,Base.ViewIndex}, dims::NTuple{M,Integer}) where {T,N,M}
    offset = Base.compute_offset1(a, 1, I) * sizeof(T)

    refcount = a.storage.refcount[]
    @assert refcount != 0
    if refcount > 0
      Threads.atomic_add!(a.storage.refcount, 1)
    end

    b = CuArray{T,M}(a.storage, dims; a.maxsize, offset=a.offset+offset)
    if refcount > 0
        finalizer(unsafe_finalize!, b)
    end
    return b
end

@inline function unsafe_view(A, I, ::NonContiguous)
    Base.unsafe_view(Base._maybe_reshape_parent(A, Base.index_ndims(I...)), I...)
end

# pointer conversions
## contiguous
function Base.unsafe_convert(::Type{CuPtr{T}}, V::SubArray{T,N,P,<:Tuple{Vararg{Base.RangeIndex}}}) where {T,N,P}
    return Base.unsafe_convert(CuPtr{T}, parent(V)) +
           Base._memory_offset(V.parent, map(first, V.indices)...)
end
## reshaped
function Base.unsafe_convert(::Type{CuPtr{T}}, V::SubArray{T,N,P,<:Tuple{Vararg{Union{Base.RangeIndex,Base.ReshapedUnitRange}}}}) where {T,N,P}
   return Base.unsafe_convert(CuPtr{T}, parent(V)) +
          (Base.first_index(V)-1)*sizeof(T)
end


## PermutedDimsArray

Base.unsafe_convert(::Type{CuPtr{T}}, A::PermutedDimsArray) where {T} =
    Base.unsafe_convert(CuPtr{T}, parent(A))


## reshape

function Base.reshape(a::CuArray{T,M}, dims::NTuple{N,Int}) where {T,N,M}
  if prod(dims) != length(a)
      throw(DimensionMismatch("new dimensions $(dims) must be consistent with array size $(size(a))"))
  end

  if N == M && dims == size(a)
      return a
  end

  _derived_array(T, N, a, dims)
end

# create a derived array (reinterpreted or reshaped) that's still a CuArray
@inline function _derived_array(::Type{T}, N::Int, a::CuArray, osize::Dims) where {T}
  refcount = a.storage.refcount[]
  @assert refcount != 0
  if refcount > 0
    Threads.atomic_add!(a.storage.refcount, 1)
  end

  b = CuArray{T,N}(a.storage, osize; a.maxsize, a.offset)
  if refcount > 0
      finalizer(unsafe_finalize!, b)
  end
  return b
end


## reinterpret

function Base.reinterpret(::Type{T}, a::CuArray{S,N}) where {T,S,N}
  err = _reinterpret_exception(T, a)
  err === nothing || throw(err)

  if sizeof(T) == sizeof(S) # for N == 0
    osize = size(a)
  else
    isize = size(a)
    size1 = div(isize[1]*sizeof(S), sizeof(T))
    osize = tuple(size1, Base.tail(isize)...)
  end

  return _derived_array(T, N, a, osize)
end

function _reinterpret_exception(::Type{T}, a::AbstractArray{S,N}) where {T,S,N}
  if !isbitstype(T) || !isbitstype(S)
    return _CuReinterpretBitsTypeError{T,typeof(a)}()
  end
  if N == 0 && sizeof(T) != sizeof(S)
    return _CuReinterpretZeroDimError{T,typeof(a)}()
  end
  if N != 0 && sizeof(S) != sizeof(T)
      ax1 = axes(a)[1]
      dim = length(ax1)
      if Base.rem(dim*sizeof(S),sizeof(T)) != 0
        return _CuReinterpretDivisibilityError{T,typeof(a)}(dim)
      end
      if first(ax1) != 1
        return _CuReinterpretFirstIndexError{T,typeof(a),typeof(ax1)}(ax1)
      end
  end
  return nothing
end

struct _CuReinterpretBitsTypeError{T,A} <: Exception end
function Base.showerror(io::IO, ::_CuReinterpretBitsTypeError{T, <:AbstractArray{S}}) where {T, S}
  print(io, "cannot reinterpret an `$(S)` array to `$(T)`, because not all types are bitstypes")
end

struct _CuReinterpretZeroDimError{T,A} <: Exception end
function Base.showerror(io::IO, ::_CuReinterpretZeroDimError{T, <:AbstractArray{S,N}}) where {T, S, N}
  print(io, "cannot reinterpret a zero-dimensional `$(S)` array to `$(T)` which is of a different size")
end

struct _CuReinterpretDivisibilityError{T,A} <: Exception
  dim::Int
end
function Base.showerror(io::IO, err::_CuReinterpretDivisibilityError{T, <:AbstractArray{S,N}}) where {T, S, N}
  dim = err.dim
  print(io, """
      cannot reinterpret an `$(S)` array to `$(T)` whose first dimension has size `$(dim)`.
      The resulting array would have non-integral first dimension.
      """)
end

struct _CuReinterpretFirstIndexError{T,A,Ax1} <: Exception
  ax1::Ax1
end
function Base.showerror(io::IO, err::_CuReinterpretFirstIndexError{T, <:AbstractArray{S,N}}) where {T, S, N}
  ax1 = err.ax1
  print(io, "cannot reinterpret a `$(S)` array to `$(T)` when the first axis is $ax1. Try reshaping first.")
end

## reinterpret(reshape)

function Base.reinterpret(::typeof(reshape), ::Type{T}, a::CuArray) where {T}
  N, osize = _base_check_reshape_reinterpret(T, a)
  return _derived_array(T, N, a, osize)
end

# taken from reinterpretarray.jl
# TODO: move these Base definitions out of the ReinterpretArray struct for reuse
function _base_check_reshape_reinterpret(::Type{T}, a::CuArray{S}) where {T,S}
  isbitstype(T) || throwbits(S, T, T)
  isbitstype(S) || throwbits(S, T, S)
  if sizeof(S) == sizeof(T)
      N = ndims(a)
      osize = size(a)
  elseif sizeof(S) > sizeof(T)
      d, r = divrem(sizeof(S), sizeof(T))
      r == 0 || throwintmult(S, T)
      N = ndims(a) + 1
      osize = (d, size(a)...)
  else
      d, r = divrem(sizeof(T), sizeof(S))
      r == 0 || throwintmult(S, T)
      N = ndims(a) - 1
      N > -1 || throwsize0(S, T, "larger")
      axes(a, 1) == Base.OneTo(sizeof(T) ÷ sizeof(S)) || throwsize1(a, T)
      osize = size(a)[2:end]
  end
  return N, osize
end

@noinline function throwbits(S::Type, T::Type, U::Type)
  throw(ArgumentError("cannot reinterpret `$(S)` as `$(T)`, type `$(U)` is not a bits type"))
end

@noinline function throwintmult(S::Type, T::Type)
  throw(ArgumentError("`reinterpret(reshape, T, a)` requires that one of `sizeof(T)` (got $(sizeof(T))) and `sizeof(eltype(a))` (got $(sizeof(S))) be an integer multiple of the other"))
end

@noinline function throwsize0(S::Type, T::Type, msg)
  throw(ArgumentError("cannot reinterpret a zero-dimensional `$(S)` array to `$(T)` which is of a $msg size"))
end

@noinline function throwsize1(a::AbstractArray, T::Type)
    throw(ArgumentError("`reinterpret(reshape, $T, a)` where `eltype(a)` is $(eltype(a)) requires that `axes(a, 1)` (got $(axes(a, 1))) be equal to 1:$(sizeof(T) ÷ sizeof(eltype(a))) (from the ratio of element sizes)"))
end


## resizing

"""
  resize!(a::CuVector, n::Int)

Resize `a` to contain `n` elements. If `n` is smaller than the current collection length,
the first `n` elements will be retained. If `n` is larger, the new elements are not
guaranteed to be initialized.

Note that this operation is only supported on managed buffers, i.e., not on arrays that are
created by `unsafe_wrap` with `own=false`.
"""
function Base.resize!(A::CuVector{T}, n::Int) where T
  # TODO: add additional space to allow for quicker resizing
  maxsize = n * sizeof(T)
  bufsize = if Base.isbitsunion(T)
    # type tag array past the data
    maxsize + n
  else
    maxsize
  end

  new_storage = @context! A.storage.ctx begin
    buf = alloc(typeof(A.storage.buffer), bufsize)
    ptr = convert(CuPtr{T}, buf)
    m = Base.min(length(A), n)
    unsafe_copyto!(ptr, pointer(A), m)
    ArrayStorage(buf, A.storage.ctx, 1)
  end

  unsafe_free!(A)
  A.storage = new_storage
  A.dims = (n,)
  A.maxsize = maxsize
  A.offset = 0

  A
end
