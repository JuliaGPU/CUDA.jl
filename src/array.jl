export CuArray, CuVector, CuMatrix, CuVecOrMat, cu, is_device, is_unified, is_host


## array type

function hasfieldcount(@nospecialize(dt))
    try
        fieldcount(dt)
    catch
        return false
    end
    return true
end

explain_nonisbits(@nospecialize(T), depth=0) = "  "^depth * "$T is not a bitstype\n"

function explain_eltype(@nospecialize(T), depth=0; maxdepth=10)
    depth > maxdepth && return ""

    if T isa Union
      msg = "  "^depth * "$T is a union that's not allocated inline\n"
      for U in Base.uniontypes(T)
        if !Base.allocatedinline(U)
          msg *= explain_eltype(U, depth+1)
        end
      end
    elseif Base.ismutabletype(T) && Base.datatype_fieldcount(T) != 0
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
#    these are simply stored contiguously in memory.
# 2. structs of unions (`struct Foo; x::Union{Int, Float32}; end`)
#    these are stored with a selector at the end (handled by Julia).
# 3. bitstype unions (`Union{Int, Float32}`, etc)
#    these are stored contiguously and require a selector array (handled by us)
# As well as "mutable singleton" types like `Symbol` that use pointer-identity

function valid_type(@nospecialize(T))
  if Base.allocatedinline(T)
    if hasfieldcount(T)
      return all(valid_type, fieldtypes(T))
    end
    return true
  elseif Base.ismutabletype(T)
    return Base.datatype_fieldcount(T) == 0
  end
  return false
end

@inline function check_eltype(name, T)                      
  if !valid_type(T) 
    explanation = explain_eltype(T)
    error("""
      $name only supports element types that are allocated inline.
      $explanation""")
  end
end

mutable struct CuArray{T,N,M} <: AbstractGPUArray{T,N}
  data::DataRef{Managed{M}}

  maxsize::Int  # maximum data size; excluding any selector bytes
  offset::Int   # offset of the data in memory, in number of elements

  dims::Dims{N}

  function CuArray{T,N,M}(::UndefInitializer, dims::Dims{N}) where {T,N,M}
    check_eltype("CuArray", T)
    maxsize = prod(dims) * aligned_sizeof(T)
    bufsize = if Base.isbitsunion(T)
      # type tag array past the data
      maxsize + prod(dims)
    else
      maxsize
    end

    data = GPUArrays.cached_alloc((CuArray, device(), M, bufsize)) do
        DataRef(pool_free, pool_alloc(M, bufsize))
    end
    obj = new{T,N,M}(data, maxsize, 0, dims)
    finalizer(unsafe_free!, obj)
    return obj
  end

  function CuArray{T,N}(data::DataRef{Managed{M}}, dims::Dims{N};
                        maxsize::Int=prod(dims) * aligned_sizeof(T), offset::Int=0) where {T,N,M}
    check_eltype("CuArray", T)
    obj = new{T,N,M}(data, maxsize, offset, dims)
    finalizer(unsafe_free!, obj)
    return obj
  end
end

GPUArrays.storage(a::CuArray) = a.data


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

# unspecified memory allocation
const default_memory = let str = Preferences.@load_preference("default_memory", "device")
  if str == "device"
    DeviceMemory
  elseif str == "unified"
    UnifiedMemory
  elseif str == "host"
    HostMemory
  else
    error("unknown default memory type: $default_memory")
  end
end
CuArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N} =
  CuArray{T,N,default_memory}(undef, dims)

# memory, type and dimensionality specified
CuArray{T,N,M}(::UndefInitializer, dims::NTuple{N,Integer}) where {T,N,M} =
  CuArray{T,N,M}(undef, convert(Tuple{Vararg{Int}}, dims))
CuArray{T,N,M}(::UndefInitializer, dims::Vararg{Integer,N}) where {T,N,M} =
  CuArray{T,N,M}(undef, convert(Tuple{Vararg{Int}}, dims))

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
CuArray{T,1,M}() where {T,M} = CuArray{T,1,M}(undef, 0)
CuArray{T,1}() where {T} = CuArray{T,1}(undef, 0)

# do-block constructors
for (ctor, tvars) in (:CuArray => (),
                      :(CuArray{T}) => (:T,),
                      :(CuArray{T,N}) => (:T, :N),
                      :(CuArray{T,N,M}) => (:T, :N, :M))
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

Base.similar(a::CuArray{T,N,M}) where {T,N,M} =
  CuArray{T,N,M}(undef, size(a))
Base.similar(a::CuArray{T,<:Any,M}, dims::Base.Dims{N}) where {T,N,M} =
  CuArray{T,N,M}(undef, dims)
Base.similar(a::CuArray{<:Any,<:Any,M}, ::Type{T}, dims::Base.Dims{N}) where {T,N,M} =
  CuArray{T,N,M}(undef, dims)

function Base.copy(a::CuArray{T,N}) where {T,N}
  b = similar(a)
  @inbounds copyto!(b, a)
end

function Base.deepcopy_internal(x::CuArray, dict::IdDict)
  haskey(dict, x) && return dict[x]::typeof(x)
  return dict[x] = copy(x)
end


## unsafe_wrap

"""
  # simple case, wrapping a CuArray around an existing GPU pointer
  unsafe_wrap(CuArray, ptr::CuPtr{T}, dims; own=false, ctx=context())

  # wraps a CPU array object around a unified GPU array
  unsafe_wrap(Array, a::CuArray)

  # wraps a GPU array object around a CPU array.
  # if your system supports HMM, this is a fast operation.
  # in other cases, it has to use page locking, which can be slow.
  unsafe_wrap(CuArray, ptr::ptr{T}, dims)
  unsafe_wrap(CuArray, a::Array)

Wrap a `CuArray` object around the data at the address given by the CUDA-managed pointer
`ptr`. The element type `T` determines the array element type. `dims` is either an integer
(for a 1d array) or a tuple of the array dimensions. `own` optionally specified whether
Julia should take ownership of the memory, calling `cudaFree` when the array is no longer
referenced. The `ctx` argument determines the CUDA context where the data is allocated in.
"""
unsafe_wrap

# managed pointer to CuArray
function Base.unsafe_wrap(::Union{Type{CuArray},Type{CuArray{T}},Type{CuArray{T,N}}},
                          ptr::CuPtr{T}, dims::NTuple{N,Int};
                          own::Bool=false, ctx::CuContext=context()) where {T,N}
  # identify the memory type
  M = try
    typ = memory_type(ptr)
    if is_managed(ptr)
      UnifiedMemory
    elseif typ == CU_MEMORYTYPE_DEVICE
      DeviceMemory
    elseif typ == CU_MEMORYTYPE_HOST
      HostMemory
    else
      error("Unknown memory type; please file an issue.")
    end
  catch err
      throw(ArgumentError("Could not identify the memory type; are you passing a valid CUDA pointer to unsafe_wrap?"))
  end

  unsafe_wrap(CuArray{T,N,M}, ptr, dims; own, ctx)
end
function Base.unsafe_wrap(::Type{CuArray{T,N,M}},
                          ptr::CuPtr{T}, dims::NTuple{N,Int};
                          own::Bool=false, ctx::CuContext=context()) where {T,N,M}
  check_eltype("unsafe_wrap(CuArray, ...)", T)
  sz = prod(dims) * aligned_sizeof(T)

  # create a memory object
  mem = if M == UnifiedMemory
    UnifiedMemory(ctx, ptr, sz)
  elseif M == DeviceMemory
    # TODO: can we identify whether this pointer was allocated asynchronously?
    DeviceMemory(device(ctx), ctx, ptr, sz, false)
  elseif M == HostMemory
    HostMemory(ctx, host_pointer(ptr), sz)
  else
    throw(ArgumentError("Unknown memory type $M"))
  end

  data = DataRef(own ? pool_free : Returns(nothing), Managed(mem))
  CuArray{T,N}(data, dims)
end
# integer size input
function Base.unsafe_wrap(::Union{Type{CuArray},Type{CuArray{T}},Type{CuArray{T,1}}},
                          p::CuPtr{T}, dim::Int;
                          own::Bool=false, ctx::CuContext=context()) where {T}
  unsafe_wrap(CuArray{T,1}, p, (dim,); own, ctx)
end
function Base.unsafe_wrap(::Type{CuArray{T,1,M}}, p::CuPtr{T}, dim::Int;
                          own::Bool=false, ctx::CuContext=context()) where {T,M}
  unsafe_wrap(CuArray{T,1,M}, p, (dim,); own, ctx)
end

# managed pointer to Array
function Base.unsafe_wrap(::Union{Type{Array},Type{Array{T}},Type{Array{T,N}}},
                          p::CuPtr{T}, dims::NTuple{N,Int};
                          own::Bool=false) where {T,N}
  if !is_managed(p) && memory_type(p) != CU_MEMORYTYPE_HOST
    throw(ArgumentError("Can only create a CPU array object from a unified or host CUDA array"))
  end
  unsafe_wrap(Array{T,N}, reinterpret(Ptr{T}, p), dims; own)
end
# integer size input
function Base.unsafe_wrap(::Union{Type{Array},Type{Array{T}},Type{Array{T,1}}},
                          p::CuPtr{T}, dim::Int; own::Bool=false) where {T}
  unsafe_wrap(Array{T,1}, p, (dim,); own)
end
# array input
function Base.unsafe_wrap(::Union{Type{Array},Type{Array{T}},Type{Array{T,N}}},
                          a::CuArray{T,N}) where {T,N}
  p = pointer(a; type=HostMemory)
  unsafe_wrap(Array, p, size(a))
end

# unmanaged pointer to CuArray
supports_hmm(dev) = driver_version() >= v"12.2" &&
                    attribute(dev, DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS) == 1
function Base.unsafe_wrap(::Type{CuArray{T,N,M}}, p::Ptr{T}, dims::NTuple{N,Int};
                          ctx::CuContext=context()) where {T,N,M<:AbstractMemory}
  isbitstype(T) || throw(ArgumentError("Can only unsafe_wrap a pointer to a bits type"))
  sz = prod(dims) * aligned_sizeof(T)

  data = if M == UnifiedMemory
    # HMM extends unified memory to include system memory
    supports_hmm(device(ctx)) ||
      throw(ArgumentError("Cannot wrap system memory as unified memory on your system"))
    mem = UnifiedMemory(ctx, reinterpret(CuPtr{Nothing}, p), sz)
    DataRef(Returns(nothing), Managed(mem))
  elseif M == HostMemory
    # register as device-accessible host memory
    mem = context!(ctx) do
      register(HostMemory, p, sz, MEMHOSTREGISTER_DEVICEMAP)
    end
    DataRef(Managed(mem)) do args...
      context!(ctx; skip_destroyed=true) do
        unregister(mem)
      end
    end
  else
    throw(ArgumentError("Cannot wrap system memory as $M"))
  end

  CuArray{T,N}(data, dims)
end
function Base.unsafe_wrap(::Union{Type{CuArray},Type{CuArray{T}},Type{CuArray{T,N}}},
                          p::Ptr{T}, dims::NTuple{N,Int}; ctx::CuContext=context()) where {T,N}
  if supports_hmm(device(ctx))
    Base.unsafe_wrap(CuArray{T,N,UnifiedMemory}, p, dims; ctx)
  else
    Base.unsafe_wrap(CuArray{T,N,HostMemory}, p, dims; ctx)
  end
end
# integer size input
Base.unsafe_wrap(::Union{Type{CuArray},Type{CuArray{T}},Type{CuArray{T,1}}},
                 p::Ptr{T}, dim::Int) where {T} =
  unsafe_wrap(CuArray{T,1}, p, (dim,))
Base.unsafe_wrap(::Type{CuArray{T,1,M}}, p::Ptr{T}, dim::Int) where {T,M} =
  unsafe_wrap(CuArray{T,1,M}, p, (dim,))
# array input
Base.unsafe_wrap(::Union{Type{CuArray},Type{CuArray{T}},Type{CuArray{T,N}}},
                 a::Array{T,N}) where {T,N} =
  unsafe_wrap(CuArray{T,N}, pointer(a), size(a))
Base.unsafe_wrap(::Type{CuArray{T,N,M}}, a::Array{T,N}) where {T,N,M} =
  unsafe_wrap(CuArray{T,N,M}, pointer(a), size(a))


## array interface

Base.elsize(::Type{<:CuArray{T}}) where {T} = aligned_sizeof(T)

Base.size(x::CuArray) = x.dims
Base.sizeof(x::CuArray) = Base.elsize(x) * length(x)

context(A::CuArray) = A.data[].mem.ctx
device(A::CuArray) = device(A.data[].mem.ctx)

memory_type(x::CuArray) = memory_type(typeof(x))
memory_type(::Type{<:CuArray{<:Any,<:Any,M}}) where {M} = @isdefined(M) ? M : Any

is_device(a::CuArray) = memory_type(a) == DeviceMemory
is_unified(a::CuArray) = memory_type(a) == UnifiedMemory
is_host(a::CuArray) = memory_type(a) == HostMemory


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

"""
    pointer(::CuArray, [index=1]; [type=DeviceMemory])

Get the native address of a CUDA array object, optionally at a given location `index`.

The `type` argument indicates what kind of pointer to return, either a GPU-accessible
`CuPtr` when passing `type=DeviceMemory`, or a CPU-accessible `Ptr` when passing
`type=HostMemory`.

!!! note

    The `type` argument indicates what kind of pointer to return, i.e., where the data will
    be accessed from. This is separate from where the data is stored. For example an array
    backed by `HostMemory` may be accessed from both the CPU and GPU, so it is valid to
    pass `type=HostMemory` or `type=DeviceMemory` (but note that accessing `HostMemory` from
    the GPU is typically slow). That also implies it is not valid to pass
    `type=UnifiedMemory`, as this does not indicate where the pointer will be accessed from.
"""
@inline function Base.pointer(x::StridedCuArray{T}, i::Integer=1; type=DeviceMemory) where T
    PT = if type == DeviceMemory
      CuPtr{T}
    elseif type == HostMemory
      Ptr{T}
    else
      error("unknown memory type")
    end
    Base.unsafe_convert(PT, x) + Base._memory_offset(x, i)
end

# anything that's (secretly) backed by a CuArray
const AnyCuArray{T,N} = Union{CuArray{T,N}, WrappedArray{T,N,CuArray,CuArray{T,N}}}
const AnyCuVector{T} = AnyCuArray{T,1}
const AnyCuMatrix{T} = AnyCuArray{T,2}
const AnyCuVecOrMat{T} = Union{AnyCuVector{T}, AnyCuMatrix{T}}


## interop with other arrays

@inline function CuArray{T,N,M}(xs::AbstractArray{<:Any,N}) where {T,N,M}
  A = CuArray{T,N,M}(undef, size(xs))
  copyto!(A, convert(Array{T}, xs))
  return A
end

@inline CuArray{T,N}(xs::AbstractArray{<:Any,N}) where {T,N} =
  CuArray{T,N,default_memory}(xs)

@inline CuArray{T,N}(xs::CuArray{<:Any,N,M}) where {T,N,M} =
  CuArray{T,N,M}(xs)

# underspecified constructors
CuArray{T}(xs::AbstractArray{S,N}) where {T,N,S} = CuArray{T,N}(xs)
(::Type{CuArray{T,N} where T})(x::AbstractArray{S,N}) where {S,N} = CuArray{S,N}(x)
CuArray(A::AbstractArray{T,N}) where {T,N} = CuArray{T,N}(A)

# copy xs to match Array behavior
CuArray{T,N,M}(xs::CuArray{T,N,M}) where {T,N,M} = copy(xs)
CuArray{T,N}(xs::CuArray{T,N,M}) where {T,N,M} = copy(xs)


## conversions

Base.convert(::Type{T}, x::T) where T <: CuArray = x

# defer the conversion to Managed, where we handle memory consistency
# XXX: conversion to Memory or Managed memory by cconvert?
Base.unsafe_convert(typ::Type{Ptr{T}}, x::CuArray{T}) where {T} =
  convert(typ, x.data[]) + x.offset * Base.elsize(x)
Base.unsafe_convert(typ::Type{CuPtr{T}}, x::CuArray{T}) where {T} =
  convert(typ, x.data[]) + x.offset * Base.elsize(x)


## indexing

function Base.getindex(x::CuArray{<:Any, <:Any, <:Union{HostMemory,UnifiedMemory}}, I::Int)
  @boundscheck checkbounds(x, I)
  unsafe_load(pointer(x, I; type=HostMemory))
end

function Base.setindex!(x::CuArray{<:Any, <:Any, <:Union{HostMemory,UnifiedMemory}}, v, I::Int)
  @boundscheck checkbounds(x, I)
  unsafe_store!(pointer(x, I; type=HostMemory), v)
end


## interop with device arrays

function Base.unsafe_convert(::Type{CuDeviceArray{T,N,AS.Global}}, a::DenseCuArray{T,N}) where {T,N}
  CuDeviceArray{T,N,AS.Global}(reinterpret(LLVMPtr{T,AS.Global}, pointer(a)), size(a),
                               a.maxsize - a.offset*Base.elsize(a))
end


## synchronization

synchronize(x::CuArray) = synchronize(x.data[])

"""
    enable_synchronization!(arr::CuArray, enable::Bool)

By default `CuArray`s are implicitly synchronized when they are accessed on different CUDA
devices or streams. This may be unwanted when e.g. using disjoint slices of memory across
different tasks. This function allows to enable or disable this behavior.

!!! warning

    Disabling implicit synchronization affects _all_ `CuArray`s that are referring to the
    same underlying memory. Unsafe use of this API _will_ result in data corruption.

    This API is only provided as an escape hatch, and should not be used without careful
    consideration. If automatic synchronization is generally problematic for your use case,
    it is recommended to figure out a better model instead and file an issue or pull request.
    For more details see [this discussion](https://github.com/JuliaGPU/CUDA.jl/issues/2617).
"""
function enable_synchronization!(arr::CuArray, enable::Bool=true)
    arr.data[].synchronizing = enable
    return arr
end


## memory copying

if VERSION >= v"1.11.0-DEV.753"
function typetagdata(a::Array, i=1)
  ptr_or_offset = Int(a.ref.ptr_or_offset)
  @ccall(jl_genericmemory_typetagdata(a.ref.mem::Any)::Ptr{UInt8}) + ptr_or_offset + i - 1
end
else
typetagdata(a::Array, i=1) = ccall(:jl_array_typetagdata, Ptr{UInt8}, (Any,), a) + i - 1
end
function typetagdata(a::CuArray, i=1; type=DeviceMemory)
  PT = if type == DeviceMemory
    CuPtr{UInt8}
  elseif type == HostMemory
    Ptr{UInt8}
  else
    error("unknown memory type")
  end
  convert(PT, a.data[]) + a.maxsize + a.offset + i - 1
end

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

# NOTE: we only switch contexts here to avoid illegal memory accesses.
# our current programming model expects users to manage the active device.

function Base.unsafe_copyto!(dest::DenseCuArray{T}, doffs,
                             src::Array{T}, soffs, n) where T
  context!(context(dest)) do
    # the copy below may block in `libcuda`, so it'd be good to perform a nonblocking
    # synchronization here, but the exact cases are hard to know and detect (e.g., unpinned
    # memory normally blocks, but not for all sizes, and not on all memory architectures).
    GC.@preserve src dest begin
      # semantically, it is not safe for this operation to execute asynchronously, because
      # the Array may be collected before the copy starts executing. However, when using
      # unpinned memory, CUDA first stages a copy to a pinned buffer that will outlive
      # the source array, making this operation safe.
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
    # see comment above; this copy may also block in `libcuda` when dealing with e.g.
    # unpinned memory, but even more likely because we need to wait for the GPU to finish
    # so that the expected data is available. because of that, eagerly perform a nonblocking
    # synchronization first as to maximize the time spent executing Julia code.
    synchronize(src)

    GC.@preserve src dest begin
      unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n; async=false)
      if Base.isbitsunion(T)
        unsafe_copyto!(typetagdata(dest, doffs), typetagdata(src, soffs), n; async=false)
      end
    end
  end
  return dest
end

function Base.unsafe_copyto!(dest::DenseCuArray{T}, doffs,
                             src::DenseCuArray{T}, soffs, n) where T
  if device(src) == device(dest) ||
     maybe_enable_peer_access(device(src), device(dest)) == 1
    # use direct device-to-device copy
    context!(context(src)) do
      GC.@preserve src dest begin
        unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n; async=true)
        if Base.isbitsunion(T)
          unsafe_copyto!(typetagdata(dest, doffs), typetagdata(src, soffs), n; async=true)
        end
      end
    end
  else
    # stage through host memory
    tmp = Vector{T}(undef, n)
    unsafe_copyto!(tmp, 1, src, soffs, n)
    unsafe_copyto!(dest, doffs, tmp, 1, n)
  end
  return dest
end

# optimization: memcpy on the CPU for Array <-> unified or host arrays

# NOTE: synchronization is best-effort, since we don't keep track of the
#       dependencies and streams using each array backed by unified memory.

function Base.unsafe_copyto!(dest::DenseCuArray{T,<:Any,<:Union{UnifiedMemory,HostMemory}}, doffs,
                             src::Array{T}, soffs, n) where T
  # maintain stream-ordered semantics: even though the pointer conversion should sync when
  # needed, it's possible that misses captured memory, so ensure copying is always correct.
  synchronize(dest)

  GC.@preserve src dest begin
    ptr = pointer(src, soffs)
    unsafe_copyto!(pointer(dest, doffs; type=HostMemory), ptr, n)
    if Base.isbitsunion(T)
      ptr = typetagdata(src, soffs)
      unsafe_copyto!(typetagdata(dest, doffs; type=HostMemory), ptr, n)
    end
  end
  return dest
end

function Base.unsafe_copyto!(dest::Array{T}, doffs,
                             src::DenseCuArray{T,<:Any,<:Union{UnifiedMemory,HostMemory}}, soffs, n) where T
  # maintain stream-ordered semantics: even though the pointer conversion should sync when
  # needed, it's possible that misses captured memory, so ensure copying is always correct.
  synchronize(src)

  GC.@preserve src dest begin
    ptr = pointer(dest, doffs)
    unsafe_copyto!(ptr, pointer(src, soffs; type=HostMemory), n)
    if Base.isbitsunion(T)
      ptr = typetagdata(dest, doffs)
      unsafe_copyto!(ptr, typetagdata(src, soffs; type=HostMemory), n)
    end
  end

  return dest
end

# optimization: memcpy between host or unified arrays without context switching

function Base.unsafe_copyto!(dest::DenseCuArray{T,<:Any,<:Union{UnifiedMemory,HostMemory}}, doffs,
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
                             src::DenseCuArray{T,<:Any,<:Union{UnifiedMemory,HostMemory}}, soffs, n) where T
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

function Base.unsafe_copyto!(dest::DenseCuArray{T,<:Any,<:Union{UnifiedMemory,HostMemory}}, doffs,
                             src::DenseCuArray{T,<:Any,<:Union{UnifiedMemory,HostMemory}}, soffs, n) where T
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
Adapt.adapt_storage(::Type{<:CuArray{T, N, M}}, xs::AT) where {T, N, M, AT<:AbstractArray} =
  isbitstype(AT) ? xs : convert(CuArray{T,N,M}, xs)


## opinionated gpu array adaptor

# eagerly converts Float64 to Float32, for performance reasons

struct CuArrayKernelAdaptor{M} end

Adapt.adapt_storage(::CuArrayKernelAdaptor{M}, xs::AbstractArray{T,N}) where {T,N,M} =
  isbits(xs) ? xs : CuArray{T,N,M}(xs)

Adapt.adapt_storage(::CuArrayKernelAdaptor{M}, xs::AbstractArray{T,N}) where {T<:AbstractFloat,N,M} =
  isbits(xs) ? xs : CuArray{Float32,N,M}(xs)

Adapt.adapt_storage(::CuArrayKernelAdaptor{M}, xs::AbstractArray{T,N}) where {T<:Complex{<:AbstractFloat},N,M} =
  isbits(xs) ? xs : CuArray{ComplexF32,N,M}(xs)

# not for Float16
Adapt.adapt_storage(::CuArrayKernelAdaptor{M}, xs::AbstractArray{T,N}) where {T<:Union{Float16,BFloat16},N,M} =
  isbits(xs) ? xs : CuArray{T,N,M}(xs)

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
1×3 adjoint(::CuArray{Float32, 1, CUDA.DeviceMemory}) with eltype Float32:
 1.0  1.0  1.0

julia> cu(zeros(1, 3); unified=true)
1×3 CuArray{Float32, 2, CUDA.UnifiedMemory}:
 0.0  0.0  0.0

julia> cu(1:3)
1:3

julia> CuArray(ones(3)')  # ignores Adjoint, preserves Float64
1×3 CuArray{Float64, 2, CUDA.DeviceMemory}:
 1.0  1.0  1.0

julia> adapt(CuArray, ones(3)')  # this restores Adjoint wrapper
1×3 adjoint(::CuArray{Float64, 1, CUDA.DeviceMemory}) with eltype Float64:
 1.0  1.0  1.0

julia> CuArray(1:3)
3-element CuArray{Int64, 1, CUDA.DeviceMemory}:
 1
 2
 3
```
"""
@inline function cu(xs; device::Bool=false, unified::Bool=false, host::Bool=false)
  if device + unified + host > 1
    throw(ArgumentError("Can only specify one of `device`, `unified`, or `host`"))
  end
  memory = if device
    DeviceMemory
  elseif unified
    UnifiedMemory
  elseif host
    HostMemory
  else
    default_memory
  end
  adapt(CuArrayKernelAdaptor{memory}(), xs)
end

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
    GC.@preserve A memset(convert(CuPtr{U}, pointer(A)), y, length(A))
  end
  A
end


## derived arrays

function GPUArrays.derive(::Type{T}, a::CuArray, dims::Dims{N}, offset::Int) where {T,N}
  offset = (a.offset * Base.elsize(a)) ÷ aligned_sizeof(T) + offset
  CuArray{T,N}(copy(a.data), dims; a.maxsize, offset)
end


## views

# pointer conversions
function Base.unsafe_convert(::Type{CuPtr{T}}, V::SubArray{T,N,P,<:Tuple{Vararg{Base.RangeIndex}}}) where {T,N,P}
    return Base.unsafe_convert(CuPtr{T}, parent(V)) +
           Base._memory_offset(V.parent, map(first, V.indices)...)
end
function Base.unsafe_convert(::Type{CuPtr{T}}, V::SubArray{T,N,P,<:Tuple{Vararg{Union{Base.RangeIndex,Base.ReshapedUnitRange}}}}) where {T,N,P}
   return Base.unsafe_convert(CuPtr{T}, parent(V)) +
          (Base.first_index(V)-1)*aligned_sizeof(T)
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
  n == length(A) && return A

  # TODO: add additional space to allow for quicker resizing
  maxsize = n * aligned_sizeof(T)
  bufsize = if isbitstype(T)
    maxsize
  else
    # type tag array past the data
    maxsize + n
  end

  # replace the data with a new one. this 'unshares' the array.
  # as a result, we can safely support resizing unowned buffers.
  new_data = context!(context(A)) do
    mem = pool_alloc(memory_type(A), bufsize)
    ptr = convert(CuPtr{T}, mem)
    m = min(length(A), n)
    if m > 0
      GC.@preserve A unsafe_copyto!(ptr, pointer(A), m)
    end
    DataRef(pool_free, mem)
  end
  unsafe_free!(A)

  A.data = new_data
  A.dims = (n,)
  A.maxsize = maxsize
  A.offset = 0

  A
end

# new version of resizing
function new_resize!(A::CuVector{T}, n::Integer) where T
  n == length(A) && return A

  # how to better choose the new size?
  if n > length(A) || n < length(A) / 2
    len = n > length(A) ? max(n, 2 * length(A)) : n

    maxsize = len * aligned_sizeof(T)
	  bufsize = if isbitstype(T)
    		maxsize
  	else
    	# type tag array past the data
   	 	maxsize + len
  	end

    new_data = context!(context(A)) do
      mem = pool_alloc(memory_type(A), bufsize)
      ptr = convert(CuPtr{T}, mem)
      m = min(length(A), n)
      if m > 0
        GC.@preserve A unsafe_copyto!(ptr, pointer(A), m)
      end
      DataRef(pool_free, mem)
    end
    unsafe_free!(A)
    A.data = new_data
    A.maxsize = maxsize
    A.offset = 0
  end

  A.dims = (n,)
  A
end