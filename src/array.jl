export CuArray, CuVector, CuMatrix, CuVecOrMat, cu, is_unified


## array type

function hasfieldcount(@nospecialize(dt))
    try
        fieldcount(dt)
    catch
        return false
    end
    return true
end

function explain_eltype(@nospecialize(T), depth=0; maxdepth=10)
    depth > maxdepth && return ""

    if T isa Union
      msg = "  "^depth * "$T is a union that's not allocated inline\n"
      for U in Base.uniontypes(T)
        if !Base.allocatedinline(U)
          msg *= explain_eltype(U, depth+1)
        end
      end
    elseif Base.ismutabletype(T)
      msg = "  "^depth * "$T is a mutable type\n"
    elseif hasfieldcount(T)
      msg = "  "^depth * "$T is a struct that's not allocated inline\n"
      for U in fieldtypes(T)
          if !Base.allocatedinline(U)
              msg *= explain_nonisbits(U, depth+1)
          end
      end
    else
      msg = "  "^depth * "$T is not allocated inline\n"
    end
    return msg
end

# CuArray only supports element types that are allocated inline (`Base.allocatedinline`).
# These come in three forms:
# 1. plain bitstypes (`Int`, `(Float32, Float64)`, plain immutable structs, etc).
#    these are simply stored contiguously in the buffer.
# 2. structs of unions (`struct Foo; x::Union{Int, Float32}; end`)
#    these are stored with a selector at the end (handled by Julia).
# 3. bitstype unions (`Union{Int, Float32}`, etc)
#    these are stored contiguously and require a selector array (handled by us)
function check_eltype(T)
  if !Base.allocatedinline(T)
    explanation = explain_eltype(T)
    error("""
      CuArray only supports element types that are allocated inline.
      $explanation""")
  end
end

mutable struct CuArray{T,N,B} <: AbstractGPUArray{T,N}
  data::DataRef{B}

  maxsize::Int  # maximum data size; excluding any selector bytes
  offset::Int   # offset of the data in the buffer, in number of elements

  dims::Dims{N}

  function CuArray{T,N,B}(::UndefInitializer, dims::Dims{N}) where {T,N,B}
    check_eltype(T)
    maxsize = prod(dims) * sizeof(T)
    bufsize = if Base.isbitsunion(T)
      # type tag array past the data
      maxsize + prod(dims)
    else
      maxsize
    end
    buf = alloc(B, bufsize)
    data = DataRef(_free_buffer, buf)
    obj = new{T,N,B}(data, maxsize, 0, dims)
    finalizer(unsafe_finalize!, obj)
  end

  function CuArray{T,N}(data::DataRef{B}, dims::Dims{N};
                        maxsize::Int=prod(dims) * sizeof(T), offset::Int=0) where {T,N,B}
    check_eltype(T)
    obj = new{T,N,B}(copy(data), maxsize, offset, dims)
    finalizer(unsafe_finalize!, obj)
  end
end

function _free_buffer(buf, early)
  context!(buf.ctx; skip_destroyed=true) do
    # during task or process finalization, the local stream might be destroyed already, so
    # use the default stream. additionally, since we don't use per-thread APIs, this default
    # stream follows legacy semantics and will synchronize all other streams. this protects
    # against freeing resources that are still in use.
    #
    # TODO: although this is still an asynchronous operation, even when using the default
    # stream, it synchronizes "too much". we could do better, e.g., by keeping track of all
    # streams involved, or by refcounting uses and decrementing that refcount after the
    # operation using `cuLaunchHostFunc`. See CUDA.jl#778 and CUDA.jl#780 for details.
    s = early ? stream() : default_stream()

    free(buf; stream=s)
  end
end

"""
    CUDA.unsafe_free!(a::CuArray)

Release the memory of an array for reuse by future allocations. This operation is
performed automatically by the GC when an array goes out of scope, but can be called
earlier to reduce pressure on the memory allocator.
"""
unsafe_free!(xs::CuArray) = GPUArrays.unsafe_free!(xs.data, true)
unsafe_finalize!(xs::CuArray) = GPUArrays.unsafe_free!(xs.data, false)


## alias detection

Base.dataids(A::CuArray) = (UInt(pointer(A)),)

Base.unaliascopy(A::CuArray) = copy(A)

function Base.mightalias(A::CuArray, B::CuArray)
  rA = pointer(A):pointer(A)+sizeof(A)
  rB = pointer(B):pointer(B)+sizeof(B)
  return first(rA) <= first(rB) < last(rA) || first(rB) <= first(rA) < last(rB)
end


## convenience constructors

const CuVector{T} = CuArray{T,1}
const CuMatrix{T} = CuArray{T,2}
const CuVecOrMat{T} = Union{CuVector{T},CuMatrix{T}}

# default to non-unified memory
CuArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N} =
  CuArray{T,N,Mem.DeviceBuffer}(undef, dims)
is_unified(a::CuArray) = isa(a.data[], Mem.UnifiedBuffer)

# buffer, type and dimensionality specified
CuArray{T,N,B}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N,B} =
  CuArray{T,N,B}(undef, convert(Tuple{Vararg{Int}}, dims))
CuArray{T,N,B}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N,B} =
  CuArray{T,N,B}(undef, convert(Tuple{Vararg{Int}}, dims))

# type and dimensionality specified
CuArray{T,N}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N} =
  CuArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))
CuArray{T,N}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N} =
  CuArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))

# type but not dimensionality specified
CuArray{T}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N} =
  CuArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))
CuArray{T}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N} =
  CuArray{T,N}(undef, convert(Tuple{Vararg{Int}}, dims))

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

function Base.deepcopy_internal(x::CuArray, dict::IdDict)
  haskey(dict, x) && return dict[x]::typeof(x)
  return dict[x] = copy(x)
end


"""
  unsafe_wrap(CuArray, ptr::CuPtr{T}, dims; own=false, ctx=context())

Wrap a `CuArray` object around the data at the address given by `ptr`. The pointer
element type `T` determines the array element type. `dims` is either an integer (for a 1d
array) or a tuple of the array dimensions. `own` optionally specified whether Julia should
take ownership of the memory, calling `cudaFree` when the array is no longer referenced. The
`ctx` argument determines the CUDA context where the data is allocated in.
"""
function Base.unsafe_wrap(::Union{Type{CuArray},Type{CuArray{T}},Type{CuArray{T,N}}},
                          ptr::CuPtr{T}, dims::NTuple{N,Int};
                          own::Bool=false, ctx::CuContext=context()) where {T,N}
  buf = _unsafe_wrap(T, ptr, dims; own, ctx)
  data = DataRef(own ? _free_buffer : (args...) -> (#= do nothing =#), buf)
  CuArray{T, length(dims)}(data, dims)
end
function Base.unsafe_wrap(::Type{CuArray{T,N,B}},
                          ptr::CuPtr{T}, dims::NTuple{N,Int};
                          own::Bool=false, ctx::CuContext=context()) where {T,N,B}
  buf = _unsafe_wrap(T, ptr, dims; own, ctx)
  if typeof(buf) !== B
    error("Declared buffer type does not match inferred buffer type.")
  end
  data = DataRef(own ? _free_buffer : (args...) -> (#= do nothing =#), buf)
  CuArray{T, length(dims)}(data, dims)
end

function _unsafe_wrap(::Type{T}, ptr::CuPtr{T}, dims::NTuple{N,Int};
                      own::Bool=false, ctx::CuContext=context()) where {T,N}
  isbitstype(T) || error("Can only unsafe_wrap a pointer to a bits type")
  sz = prod(dims) * sizeof(T)

  # identify the buffer
  buf = try
    typ = memory_type(ptr)
    if is_managed(ptr)
      Mem.UnifiedBuffer(ctx, ptr, sz)
    elseif typ == CU_MEMORYTYPE_DEVICE
      # TODO: can we identify whether this pointer was allocated asynchronously?
      Mem.DeviceBuffer(ctx, ptr, sz, false)
    elseif typ == CU_MEMORYTYPE_HOST
      Mem.HostBuffer(ctx, host_pointer(ptr), sz)
    else
      error("Unknown memory type; please file an issue.")
    end
  catch err
      error("Could not identify the buffer type; are you passing a valid CUDA pointer to unsafe_wrap?")
  end
  return buf
end

function Base.unsafe_wrap(Atype::Union{Type{CuArray},Type{CuArray{T}},Type{CuArray{T,1}}},
                          p::CuPtr{T}, dim::Int;
                          own::Bool=false, ctx::CuContext=context()) where {T}
  unsafe_wrap(Atype, p, (dim,); own, ctx)
end
function Base.unsafe_wrap(Atype::Type{CuArray{T,1,B}},
                          p::CuPtr{T}, dim::Int;
                          own::Bool=false, ctx::CuContext=context()) where {T,B}
  unsafe_wrap(Atype, p, (dim,); own, ctx)
end

Base.unsafe_wrap(T::Type{<:CuArray}, ::Ptr, dims::NTuple{N,Int}; kwargs...) where {N} =
  throw(ArgumentError("cannot wrap a CPU pointer with a $T"))


## array interface

Base.elsize(::Type{<:CuArray{T}}) where {T} = sizeof(T)

Base.size(x::CuArray) = x.dims
Base.sizeof(x::CuArray) = Base.elsize(x) * length(x)

context(A::CuArray) = A.data[].ctx
device(A::CuArray) = device(A.data[].ctx)


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
const DenseCuArray{T,N} = CuArray{T,N}
const DenseCuVector{T} = DenseCuArray{T,1}
const DenseCuMatrix{T} = DenseCuArray{T,2}
const DenseCuVecOrMat{T} = Union{DenseCuVector{T}, DenseCuMatrix{T}}
# XXX: these dummy aliases (DenseCuArray=CuArray) break alias printing, as
#      `Base.print_without_params` only handles the case of a single alias.

# strided arrays
const StridedSubCuArray{T,N,I<:Tuple{Vararg{Union{Base.RangeIndex, Base.ReshapedUnitRange,
                                            Base.AbstractCartesianIndex}}}} =
  SubArray{T,N,<:CuArray,I}
const StridedCuArray{T,N} = Union{CuArray{T,N}, StridedSubCuArray{T,N}}
const StridedCuVector{T} = StridedCuArray{T,1}
const StridedCuMatrix{T} = StridedCuArray{T,2}
const StridedCuVecOrMat{T} = Union{StridedCuVector{T}, StridedCuMatrix{T}}

Base.pointer(x::StridedCuArray{T}) where {T} = Base.unsafe_convert(CuPtr{T}, x)
@inline function Base.pointer(x::StridedCuArray{T}, i::Integer) where T
    Base.unsafe_convert(CuPtr{T}, x) + Base._memory_offset(x, i)
end

# anything that's (secretly) backed by a CuArray
const AnyCuArray{T,N} = Union{CuArray{T,N}, WrappedArray{T,N,CuArray,CuArray{T,N}}}
const AnyCuVector{T} = AnyCuArray{T,1}
const AnyCuMatrix{T} = AnyCuArray{T,2}
const AnyCuVecOrMat{T} = Union{AnyCuVector{T}, AnyCuMatrix{T}}


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

# copy xs to match Array behavior
CuArray{T,N,B}(xs::CuArray{T,N,B}) where {T,N,B} = copy(xs)
CuArray{T,N}(xs::CuArray{T,N,B}) where {T,N,B} = copy(xs)


## conversions

Base.convert(::Type{T}, x::T) where T <: CuArray = x


## interop with C libraries

Base.unsafe_convert(::Type{Ptr{T}}, x::CuArray{T}) where {T} =
  throw(ArgumentError("cannot take the CPU address of a $(typeof(x))"))
function Base.unsafe_convert(::Type{CuPtr{T}}, x::CuArray{T}) where {T}
  convert(CuPtr{T}, x.data[]) + x.offset*Base.elsize(x)
end


## interop with device arrays

function Base.unsafe_convert(::Type{CuDeviceArray{T,N,AS.Global}}, a::DenseCuArray{T,N}) where {T,N}
  CuDeviceArray{T,N,AS.Global}(reinterpret(LLVMPtr{T,AS.Global}, pointer(a)), size(a),
                               a.maxsize - a.offset*Base.elsize(a))
end


## memory copying

typetagdata(a::Array, i=1) = ccall(:jl_array_typetagdata, Ptr{UInt8}, (Any,), a) + i - 1
typetagdata(a::CuArray, i=1) =
  convert(CuPtr{UInt8}, a.data[]) + a.maxsize + a.offset + i - 1

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

# general case: use CUDA APIs

# NOTE: we only switch contexts here to avoid illegal memory accesses. synchronization is
#       best-effort, since we don't keep track of streams using each array.

function Base.unsafe_copyto!(dest::DenseCuArray{T}, doffs,
                             src::Array{T}, soffs, n) where T
  context!(context(dest)) do
    # operations on unpinned memory cannot be executed asynchronously, and synchronize
    # without yielding back to the Julia scheduler. prevent that by eagerly synchronizing.
    if use_nonblocking_synchronization
      is_pinned(pointer(src)) || nonblocking_synchronize(stream())
    end

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
                             src::DenseCuArray{T}, soffs, n) where T
  context!(context(src)) do
    # operations on unpinned memory cannot be executed asynchronously, and synchronize
    # without yielding back to the Julia scheduler. prevent that by eagerly synchronizing.
    if use_nonblocking_synchronization
      is_pinned(pointer(dest)) || nonblocking_synchronize(stream())
    end

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

function Base.unsafe_copyto!(dest::DenseCuArray{T}, doffs,
                             src::DenseCuArray{T}, soffs, n) where T
  context!(context(src)) do
    GC.@preserve src dest begin
      unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n; async=true)
      if Base.isbitsunion(T)
        unsafe_copyto!(typetagdata(dest, doffs), typetagdata(src, soffs), n; async=true)
      end
    end
  end
  return dest
end

# optimization: memcpy on the CPU for Array <-> unified or host arrays

# NOTE: synchronization is best-effort, since we don't keep track of the
#       dependencies and streams using each array backed by unified memory.

function Base.unsafe_copyto!(dest::DenseCuArray{T,<:Any,<:Union{Mem.UnifiedBuffer,Mem.HostBuffer}}, doffs,
                             src::Array{T}, soffs, n) where T
  # maintain stream-ordered semantics
  # XXX: alternative, use an async CUDA memcpy if the stream isn't idle?
  synchronize()

  GC.@preserve src dest begin
    cpu_ptr = pointer(src, soffs)
    unsafe_copyto!(host_pointer(pointer(dest, doffs)), cpu_ptr, n)
    if Base.isbitsunion(T)
      cpu_ptr = typetagdata(src, soffs)
      unsafe_copyto!(host_pointer(typetagdata(dest, doffs)), cpu_ptr, n)
    end
  end
  return dest
end

function Base.unsafe_copyto!(dest::Array{T}, doffs,
                             src::DenseCuArray{T,<:Any,<:Union{Mem.UnifiedBuffer,Mem.HostBuffer}}, soffs, n) where T
  # maintain stream-ordered semantics
  synchronize()

  GC.@preserve src dest begin
    cpu_ptr = pointer(dest, doffs)
    unsafe_copyto!(cpu_ptr, host_pointer(pointer(src, soffs)), n)
    if Base.isbitsunion(T)
      cpu_ptr = typetagdata(dest, doffs)
      unsafe_copyto!(cpu_ptr, host_pointer(typetagdata(src, soffs)), n)
    end
  end

  return dest
end

# optimization: memcpy between host or unified arrays without context switching

function Base.unsafe_copyto!(dest::DenseCuArray{T,<:Any,<:Union{Mem.UnifiedBuffer,Mem.HostBuffer}}, doffs,
                             src::DenseCuArray{T}, soffs, n) where T
  context!(context(src)) do
    GC.@preserve src dest begin
      unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n; async=true)
      if Base.isbitsunion(T)
        unsafe_copyto!(typetagdata(dest, doffs), typetagdata(src, soffs), n; async=true)
      end
    end
  end
  return dest
end

function Base.unsafe_copyto!(dest::DenseCuArray{T}, doffs,
                             src::DenseCuArray{T,<:Any,<:Union{Mem.UnifiedBuffer,Mem.HostBuffer}}, soffs, n) where T
  context!(context(dest)) do
    GC.@preserve src dest begin
      unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n; async=true)
      if Base.isbitsunion(T)
        unsafe_copyto!(typetagdata(dest, doffs), typetagdata(src, soffs), n; async=true)
      end
    end
  end
  return dest
end

function Base.unsafe_copyto!(dest::DenseCuArray{T,<:Any,<:Union{Mem.UnifiedBuffer,Mem.HostBuffer}}, doffs,
                             src::DenseCuArray{T,<:Any,<:Union{Mem.UnifiedBuffer,Mem.HostBuffer}}, soffs, n) where T
  GC.@preserve src dest begin
    unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n; async=true)
    if Base.isbitsunion(T)
      unsafe_copyto!(typetagdata(dest, doffs), typetagdata(src, soffs), n; async=true)
    end
  end
  return dest
end


## regular gpu array adaptor

# We don't convert isbits types in `adapt`, since they are already
# considered GPU-compatible.

Adapt.adapt_storage(::Type{CuArray}, xs::AT) where {AT<:AbstractArray} =
  isbitstype(AT) ? xs : convert(CuArray, xs)

# if specific type parameters are specified, preserve those
Adapt.adapt_storage(::Type{<:CuArray{T}}, xs::AT) where {T, AT<:AbstractArray} =
  isbitstype(AT) ? xs : convert(CuArray{T}, xs)
Adapt.adapt_storage(::Type{<:CuArray{T, N}}, xs::AT) where {T, N, AT<:AbstractArray} =
  isbitstype(AT) ? xs : convert(CuArray{T,N}, xs)
Adapt.adapt_storage(::Type{<:CuArray{T, N, B}}, xs::AT) where {T, N, B, AT<:AbstractArray} =
  isbitstype(AT) ? xs : convert(CuArray{T,N,B}, xs)


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

"""
    cu(A; unified=false)

Opinionated GPU array adaptor, which may alter the element type `T` of arrays:
* For `T<:AbstractFloat`, it makes a `CuArray{Float32}` for performance reasons.
  (Except that `Float16` and `BFloat16` element types are not changed.)
* For `T<:Complex{<:AbstractFloat}` it makes a `CuArray{ComplexF32}`.
* For other `isbitstype(T)`, it makes a `CuArray{T}`.

By contrast, `CuArray(A)` never changes the element type.

Uses Adapt.jl to act inside some wrapper structs.

# Examples

```
julia> cu(ones(3)')
1×3 adjoint(::CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}) with eltype Float32:
 1.0  1.0  1.0

julia> cu(zeros(1, 3); unified=true)
1×3 CuArray{Float32, 2, CUDA.Mem.UnifiedBuffer}:
 0.0  0.0  0.0

julia> cu(1:3)
1:3

julia> CuArray(ones(3)')  # ignores Adjoint, preserves Float64
1×3 CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}:
 1.0  1.0  1.0

julia> adapt(CuArray, ones(3)')  # this restores Adjoint wrapper
1×3 adjoint(::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}) with eltype Float64:
 1.0  1.0  1.0

julia> CuArray(1:3)
3-element CuArray{Int64, 1, CUDA.Mem.DeviceBuffer}:
 1
 2
 3
```
"""
@inline cu(xs; unified::Bool=false) = adapt(CuArrayAdaptor{unified ? Mem.UnifiedBuffer : Mem.DeviceBuffer}(), xs)

Base.getindex(::typeof(cu), xs...) = CuArray([xs...])


## utilities

zeros(T::Type, dims...) = fill!(CuArray{T}(undef, dims...), zero(T))
ones(T::Type, dims...) = fill!(CuArray{T}(undef, dims...), one(T))
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
  context!(context(A)) do
    Mem.set!(convert(CuPtr{U}, pointer(A)), y, length(A))
  end
  A
end


## derived arrays

function GPUArrays.derive(::Type{T}, N::Int, a::CuArray, dims::Dims, offset::Int) where {T}
  offset = (a.offset * Base.elsize(a)) ÷ sizeof(T) + offset
  CuArray{T,N}(a.data, dims; a.maxsize, offset)
end


## views

# pointer conversions
function Base.unsafe_convert(::Type{CuPtr{T}}, V::SubArray{T,N,P,<:Tuple{Vararg{Base.RangeIndex}}}) where {T,N,P}
    return Base.unsafe_convert(CuPtr{T}, parent(V)) +
           Base._memory_offset(V.parent, map(first, V.indices)...)
end
function Base.unsafe_convert(::Type{CuPtr{T}}, V::SubArray{T,N,P,<:Tuple{Vararg{Union{Base.RangeIndex,Base.ReshapedUnitRange}}}}) where {T,N,P}
   return Base.unsafe_convert(CuPtr{T}, parent(V)) +
          (Base.first_index(V)-1)*sizeof(T)
end


## PermutedDimsArray

Base.unsafe_convert(::Type{CuPtr{T}}, A::PermutedDimsArray) where {T} =
    Base.unsafe_convert(CuPtr{T}, parent(A))


## resizing

"""
  resize!(a::CuVector, n::Integer)

Resize `a` to contain `n` elements. If `n` is smaller than the current collection length,
the first `n` elements will be retained. If `n` is larger, the new elements are not
guaranteed to be initialized.
"""
function Base.resize!(A::CuVector{T}, n::Integer) where T
  # TODO: add additional space to allow for quicker resizing
  maxsize = n * sizeof(T)
  bufsize = if isbitstype(T)
    maxsize
  else
    # type tag array past the data
    maxsize + n
  end

  # replace the data with a new one. this 'unshares' the array.
  # as a result, we can safely support resizing unowned buffers.
  new_data = context!(context(A)) do
    buf = alloc(typeof(A.data[]), bufsize)
    ptr = convert(CuPtr{T}, buf)
    m = min(length(A), n)
    if m > 0
      unsafe_copyto!(ptr, pointer(A), m)
    end
    DataRef(_free_buffer, buf)
  end
  unsafe_free!(A)

  A.data = new_data
  A.dims = (n,)
  A.maxsize = maxsize
  A.offset = 0

  A
end
