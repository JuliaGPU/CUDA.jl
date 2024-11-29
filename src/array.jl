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
#    these are simply stored contiguously in memory.
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

mutable struct CuArray{T,N,M} <: AbstractGPUArray{T,N}
  data::DataRef{Managed{M}}

  maxsize::Int  # maximum data size; excluding any selector bytes
  offset::Int   # offset of the data in memory, in number of elements

  dims::Dims{N}

  function CuArray{T,N,M}(::UndefInitializer, dims::Dims{N}) where {T,N,M}
    check_eltype(T)
    maxsize = prod(dims) * sizeof(T)
    bufsize = if Base.isbitsunion(T)
      # type tag array past the data
      maxsize + prod(dims)
    else
      maxsize
    end
    data = DataRef(pool_free, pool_alloc(M, bufsize))
    obj = new{T,N,M}(data, maxsize, 0, dims)
    finalizer(unsafe_free!, obj)
  end

  function CuArray{T,N}(data::DataRef{Managed{M}}, dims::Dims{N};
                        maxsize::Int=prod(dims) * sizeof(T), offset::Int=0) where {T,N,M}
    check_eltype(T)
    obj = new{T,N,M}(data, maxsize, offset, dims)
    finalizer(unsafe_free!, obj)
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
  isbitstype(T) || throw(ArgumentError("Can only unsafe_wrap a pointer to a bits type"))
  sz = prod(dims) * sizeof(T)

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
  sz = prod(dims) * sizeof(T)

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

Base.elsize(::Type{<:CuArray{T}}) where {T} = sizeof(T)

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


## memory copying

synchronize(x::CuArray) = synchronize(x.data[])

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

#TO DO: expand this for StridedMatrices of different shapes, currently the src needs to fit in the destination
#TO DO: add parameters doffs, soffs, n

for (destType,srcType) in ((StridedSubCuArray, SubArray) , (SubArray, StridedSubCuArray), 
                            (StridedSubCuArray, StridedSubCuArray),
                            (StridedSubCuArray, Array) ,  (Array, StridedSubCuArray), 
                            (CuArray, StridedSubCuArray) , ( StridedSubCuArray, CuArray),
                            (CuArray, SubArray) , (SubArray, CuArray) )
  @eval begin
    function Base.copyto!(dest::$destType{T,2},src::$srcType{T,2}, Copy2D::Bool=false) where {T} 
      if (dest isa StridedSubCuArray) || (dest isa SubArray)
        dest_index1=findfirst((typeof.(dest.indices) .<: Int).==0)
        dest_index2=findnext((typeof.(dest.indices) .<: Int).==0, dest_index1+1)
        dest_step_x=step(dest.indices[dest_index1])
        dest_step_height=step(dest.indices[dest_index2])
        dest_parent_size=size(parent(dest))
      else
        dest_index1=1
        dest_index2=2
        dest_step_x=1
        dest_step_height=1
        dest_parent_size=size(dest)
      end
      if (src isa StridedSubCuArray) || (src isa SubArray)
        src_index1=findfirst((typeof.(src.indices) .<: Int).==0)
        src_index2=findnext((typeof.(src.indices) .<: Int).==0, src_index1+1)
        src_step_x=step(src.indices[src_index1])
        src_step_height=step(src.indices[src_index2])
        src_parent_size=size(parent(src)) 
      else
        src_index1=1
        src_index2=2
        src_step_x=1
        src_step_height=1
        src_parent_size=size(src) 
      end

      dest_pitch1= (dest_index1==1) ? 1 :  prod(dest_parent_size[1:(dest_index1-1)])
      dest_pitch2=  prod(dest_parent_size[dest_index1:(dest_index2-1)])
      src_pitch1= (src_index1==1) ? 1 :  prod(src_parent_size[1:(src_index1-1)])
      src_pitch2= prod(src_parent_size[src_index1:(src_index2-1)])
      destLocation= ((dest isa StridedSubCuArray) || (dest isa CuArray)) ? Mem.Device : Mem.Host
      srcLocation= ((src isa StridedSubCuArray) || (src isa CuArray)) ? Mem.Device : Mem.Host
      @boundscheck checkbounds(1:size(dest, 1), 1:size(src,1))
      @boundscheck checkbounds(1:size(dest, 2), 1:size(src,2))
      
      if (size(dest,1)==size(src,1) || (Copy2D))
      #Non-contigous views can be accomodated by copy3d in certain cases
        if isinteger(src_pitch2*src_step_height/src_step_x/src_pitch1) && isinteger(dest_pitch2*dest_step_height/dest_step_x/dest_pitch1) 
          Mem.unsafe_copy3d!(pointer(dest), destLocation, pointer(src), srcLocation,
                                    1, size(src,1), size(src,2);
                                    srcPos=(1,1,1), dstPos=(1,1,1),
                                    srcPitch=src_step_x*sizeof(T)*src_pitch1,srcHeight=Int(src_pitch2*src_step_height/src_step_x/src_pitch1),
                                    dstPitch=dest_step_x*sizeof(T)*dest_pitch1, dstHeight=Int(dest_pitch2*dest_step_height/dest_step_x/dest_pitch1))
        #In other cases, use parallel threads
        else
          CUDA.synchronize()
          Base.@sync for col in 1:length(src.indices[src_index2])
            Threads.@spawn begin
              Mem.unsafe_copy3d!(pointer(view(dest,:,col)),destLocation, pointer(view(src,:,col)),  srcLocation,
                                  1, 1, size(src,1);
                                  srcPos=(1,1,1), dstPos=(1,1,1),
                                  srcPitch=sizeof(T)*src_step_x*src_pitch1,srcHeight=1,
                                  dstPitch=sizeof(T)*dest_step_x*dest_pitch1, dstHeight=1)
              CUDA.synchronize()
            end
          end
        end
      else  #Ensure same behavior as Base copying from smaller to bigger matrix if copy2D is false
        start_indices=(1:size(src,1):size(src,1)*(size(src,2)+1))
        dest_col=div.(start_indices.-1,size(dest,1)).+1
        start_indices=mod.(start_indices,size(dest,1))
        replace!(start_indices,0=>size(dest,1))
        split_col=start_indices[1:end-1].>start_indices[2:end]

        CUDA.synchronize()
        Base.@sync for col in 1:length(src.indices[src_index2])
          Threads.@spawn begin
            n= split_col[col] ? (size(dest,1)-start_indices[col]+1) : size(src,1)
            Mem.unsafe_copy3d!(pointer(view(dest,:,dest_col[col])),destLocation, pointer(view(src,:,col)),  srcLocation,
                                1, 1, n;
                                srcPos=(1,1,1), dstPos=(1,1,start_indices[col]),
                                srcPitch=sizeof(T)*src_step_x*src_pitch1,srcHeight=1,
                                dstPitch=sizeof(T)*dest_step_x*dest_pitch1, dstHeight=1)
            if split_col[col]
              Mem.unsafe_copy3d!(pointer(view(dest,:,dest_col[col]+1)),destLocation, pointer(view(src,:,col)),  srcLocation,
                                1, 1, size(src,1)-n;
                                srcPos=(1,1,n+1), dstPos=(1,1,1),
                                srcPitch=sizeof(T)*src_step_x*src_pitch1,srcHeight=1,
                                dstPitch=sizeof(T)*dest_step_x*dest_pitch1, dstHeight=1)
            end
            CUDA.synchronize()
          end
        end
      end

      return dest
    end

    function Base.copyto!(dest::$destType{T,1},doffs::Integer,src::$srcType{T,1},  soffs::Integer,
                                  n::Integer) where {T} 
      n==0 && return dest
      @boundscheck checkbounds(dest, doffs)
      @boundscheck checkbounds(dest, doffs+n-1)
      @boundscheck checkbounds(src, soffs)
      @boundscheck checkbounds(src, soffs+n-1)
      if (dest isa StridedSubCuArray) || (dest isa SubArray)
        dest_index=findfirst((typeof.(dest.indices) .<: Int).==0)
        dest_step=step(dest.indices[dest_index])
        dest_pitch=(dest_index==1) ? 1 : prod(size(parent(dest))[1:(dest_index-1)])
      else
        dest_index=1
        dest_step=1
        dest_pitch=1
      end

      if (src isa StridedSubCuArray) || (src isa SubArray)
        src_index=findfirst((typeof.(src.indices) .<: Int).==0)
        src_step=step(src.indices[src_index])
        src_pitch= (src_index==1) ? 1 : prod(size(parent(src))[1:(src_index-1)])
      else
        src_index=1
        src_step=1
        src_pitch=1
      end
      destLocation= ((dest isa StridedSubCuArray) || (dest isa CuArray)) ? Mem.Device : Mem.Host
      srcLocation= ((src isa StridedSubCuArray) || (src isa CuArray)) ? Mem.Device : Mem.Host

      Mem.unsafe_copy3d!(pointer(dest), destLocation, pointer(src), srcLocation,
                                1, 1, n;
                                srcPos=(1,1,soffs), dstPos=(1,1,doffs),
                                srcPitch=src_step*sizeof(T)*src_pitch,srcHeight=1,
                                dstPitch=dest_step*sizeof(T)*dest_pitch, dstHeight=1)
      return dest
    end



    Base.copyto!(dest::$destType{T}, src::$srcType{T}) where {T} =
      copyto!(dest, 1, src, 1, length(src))

  end
end

# general case: use CUDA APIs

# NOTE: we only switch contexts here to avoid illegal memory accesses.
# our current programming model expects users to manage the active device.

function Base.unsafe_copyto!(dest::DenseCuArray{T}, doffs,
                             src::Array{T}, soffs, n) where T
  context!(context(dest)) do
    # operations on unpinned memory cannot be executed asynchronously, and synchronize
    # without yielding back to the Julia scheduler. prevent that by eagerly synchronizing.
    if use_nonblocking_synchronization
      is_pinned(pointer(src)) || synchronize()
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
      is_pinned(pointer(dest)) || synchronize()
    end

    GC.@preserve src dest begin
      unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n; async=true)
      if Base.isbitsunion(T)
        unsafe_copyto!(typetagdata(dest, doffs), typetagdata(src, soffs), n; async=true)
      end
    end

    # users expect values to be available after this call
    synchronize(src)
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
    memset(convert(CuPtr{U}, pointer(A)), y, length(A))
  end
  A
end


## derived arrays

function GPUArrays.derive(::Type{T}, a::CuArray, dims::Dims{N}, offset::Int) where {T,N}
  offset = (a.offset * Base.elsize(a)) ÷ sizeof(T) + offset
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
    mem = alloc(memory_type(A), bufsize)
    ptr = convert(CuPtr{T}, mem)
    m = min(length(A), n)
    if m > 0
      synchronize(A)
      unsafe_copyto!(ptr, pointer(A), m)
    end
    DataRef(pool_free, Managed(mem))
  end
  unsafe_free!(A)

  A.data = new_data
  A.dims = (n,)
  A.maxsize = maxsize
  A.offset = 0

  A
end
