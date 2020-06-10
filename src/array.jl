export CuArray, CuVector, CuMatrix, CuVecOrMat, cu

mutable struct CuArray{T,N,P} <: AbstractGPUArray{T,N}
  ptr::CuPtr{T}
  dims::Dims{N}

  parent::P       # parent array (for, e.g., contiguous views keeping their parent alive)
  pooled::Bool    # is this memory backed by the memory pool?

  # for early freeing outside of the GC
  refcount::Int
  freed::Bool

  ctx::CuContext

  # constrain P
  CuArray{T,N,P}(ptr, dims, parent::Union{Nothing,CuArray}, pooled, ctx) where {T,N,P} =
    new(ptr, dims, parent, pooled, 0, false, ctx)
end

# primary array
function CuArray{T,N}(ptr::CuPtr{T}, dims::Dims{N}, pooled::Bool=true;
                      ctx=context()) where {T,N}
  self = CuArray{T,N,Nothing}(ptr, dims, nothing, pooled, ctx)
  retain(self)
  finalizer(unsafe_free!, self)
  return self
end

# derived array (e.g. view, reinterpret, ...)
function CuArray{T,N}(ptr::CuPtr{T}, dims::Dims{N}, parent::P) where {T,N,P<:CuArray}
  self = CuArray{T,N,P}(ptr, dims, parent, parent.pooled, parent.ctx)
  retain(self)
  retain(parent)
  finalizer(unsafe_free!, self)
  return self
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

Base.parent(A::CuArray{<:Any,<:Any,Nothing})     = A
Base.parent(A::CuArray{<:Any,<:Any,P}) where {P} = A.parent

Base.dataids(A::CuArray{<:Any,<:Any,Nothing})     = (UInt(pointer(A)),)
Base.dataids(A::CuArray{<:Any,<:Any,P}) where {P} = (Base.dataids(parent(A))..., UInt(pointer(A)),)

Base.unaliascopy(A::CuArray{<:Any,<:Any,Nothing}) = copy(A)
function Base.unaliascopy(A::CuArray{<:Any,<:Any,P}) where {P}
  offset = pointer(A) - pointer(A.parent)
  new_parent = Base.unaliascopy(A.parent)
  typeof(A)(pointer(new_parent) + offset, A.dims, new_parent, A.pooled, A.ctx)
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

# discard the P typevar
#
# P is just used to specialize the parent field, and does not actually affect the object,
# so we can safely discard this information when creating similar objects (`typeof(A)(...)`)
CuArray{T,N,P}(args...) where {T,N,P} = CuArray{T,N}(args...)

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
  unsafe_copyto!(ptr, pointer(a), length(a))
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
    buf = Mem.DeviceBuffer(base, prod(dims) * sizeof(T), ctx)
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

Base.pointer(x::CuArray) = x.ptr
Base.pointer(x::CuArray, i::Integer) = x.ptr + (i-1) * Base.elsize(x)


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

function Base._reshape(parent::CuArray, dims::Dims)
  n = length(parent)
  prod(dims) == n || throw(DimensionMismatch("parent has $n elements, which is incompatible with size $dims"))
  return CuArray{eltype(parent),length(dims)}(pointer(parent), dims, parent)
end
function Base._reshape(parent::CuArray{T,1}, dims::Tuple{Int}) where T
  n = length(parent)
  prod(dims) == n || throw(DimensionMismatch("parent has $n elements, which is incompatible with size $dims"))
  return parent
end


## interop with C libraries

Base.unsafe_convert(::Type{Ptr{T}}, x::CuArray{T}) where {T} = throw(ArgumentError("cannot take the CPU address of a $(typeof(x))"))
Base.unsafe_convert(::Type{Ptr{S}}, x::CuArray{T}) where {S,T} = throw(ArgumentError("cannot take the CPU address of a $(typeof(x))"))

Base.unsafe_convert(::Type{CuPtr{T}}, x::CuArray{T}) where {T} = pointer(x)
Base.unsafe_convert(::Type{CuPtr{S}}, x::CuArray{T}) where {S,T} = convert(CuPtr{S}, Base.unsafe_convert(CuPtr{T}, x))



## interop with device arrays

function Base.convert(::Type{CuDeviceArray{T,N,AS.Global}}, a::CuArray{T,N}) where {T,N}
  CuDeviceArray{T,N,AS.Global}(a.dims, DevicePtr{T,AS.Global}(pointer(a)))
end

Adapt.adapt_storage(::Adaptor, xs::CuArray{T,N}) where {T,N} =
  convert(CuDeviceArray{T,N,AS.Global}, xs)


## interop with CPU arrays

# We don't convert isbits types in `adapt`, since they are already
# considered GPU-compatible.

Adapt.adapt_storage(::Type{CuArray}, xs::AbstractArray) =
  isbits(xs) ? xs : convert(CuArray, xs)

# aggressively convert arrays of floats to float32
Adapt.adapt_storage(::Type{CuArray}, xs::AbstractArray{<:AbstractFloat}) =
  isbits(xs) ? xs : convert(CuArray{Float32}, xs)

# if an element type is specified, convert to it
Adapt.adapt_storage(::Type{<:CuArray{T}}, xs::AbstractArray) where {T} =
  isbits(xs) ? xs : convert(CuArray{T}, xs)

Adapt.adapt_storage(::Type{Array}, xs::CuArray) = convert(Array, xs)

Base.collect(x::CuArray{T,N}) where {T,N} = copyto!(Array{T,N}(undef, size(x)), x)

function Base.copyto!(dest::CuArray{T}, doffs::Integer, src::Array{T}, soffs::Integer,
                      n::Integer) where T
  @assert !dest.freed "Use of freed memory"
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  unsafe_copyto!(dest, doffs, src, soffs, n)
  return dest
end

function Base.copyto!(dest::Array{T}, doffs::Integer, src::CuArray{T}, soffs::Integer,
                      n::Integer) where T
  @assert !src.freed "Use of freed memory"
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  unsafe_copyto!(dest, doffs, src, soffs, n)
  return dest
end

function Base.copyto!(dest::CuArray{T}, doffs::Integer, src::CuArray{T}, soffs::Integer,
                      n::Integer) where T
  @assert !dest.freed && !src.freed "Use of freed memory"
  @boundscheck checkbounds(dest, doffs)
  @boundscheck checkbounds(dest, doffs+n-1)
  @boundscheck checkbounds(src, soffs)
  @boundscheck checkbounds(src, soffs+n-1)
  unsafe_copyto!(dest, doffs, src, soffs, n)
  return dest
end

function Base.unsafe_copyto!(dest::CuArray{T}, doffs, src::Array{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("Not implemented")
  end
  return dest
end

function Base.unsafe_copyto!(dest::Array{T}, doffs, src::CuArray{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n)
  if Base.isbitsunion(T)
    # copy selector bytes
    error("Not implemented")
  end
  return dest
end

function Base.unsafe_copyto!(dest::CuArray{T}, doffs, src::CuArray{T}, soffs, n) where T
  GC.@preserve src dest unsafe_copyto!(pointer(dest, doffs), pointer(src, soffs), n)
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


## utilities

cu(xs) = adapt(CuArray, xs)
Base.getindex(::typeof(cu), xs...) = CuArray([xs...])

zeros(T::Type, dims...) = fill!(CuArray{T}(undef, dims...), 0)
ones(T::Type, dims...) = fill!(CuArray{T}(undef, dims...), 1)
zeros(dims...) = zeros(Float32, dims...)
ones(dims...) = ones(Float32, dims...)
fill(v, dims...) = fill!(CuArray{typeof(v)}(undef, dims...), v)
fill(v, dims::Dims) = fill!(CuArray{typeof(v)}(undef, dims...), v)

# optimized implementation of `fill!` for types that are directly supported by memset
const MemsetTypes = Dict(1=>UInt8, 2=>UInt16, 4=>UInt32)
const MemsetCompatTypes = Union{UInt8, Int8,
                                UInt16, Int16, Float16,
                                UInt32, Int32, Float32}
function Base.fill!(A::CuArray{T}, x) where T <: MemsetCompatTypes
  U = MemsetTypes[sizeof(T)]
  y = reinterpret(U, convert(T, x))
  Mem.set!(convert(CuPtr{U}, pointer(A)), y, length(A))
  A
end


## reversing

# the kernel works by treating the array as 1d. after reversing by dimension x an element at
# pos [i1, i2, i3, ... , i{x},            ..., i{n}] will be at
# pos [i1, i2, i3, ... , d{x} - i{x} + 1, ..., i{n}] where d{x} is the size of dimension x

# out-of-place version, copying a single value per thread from input to output
function _reverse(input::CuArray{T, N}, output::CuArray{T, N}; dims::Integer=1) where {T, N}
    @assert size(input) == size(output)
    shape = [size(input)...]
    numelemsinprevdims = prod(shape[1:dims-1])
    numelemsincurrdim = shape[dims]

    function kernel(input::CuDeviceArray{T, N}, output::CuDeviceArray{T, N}) where {T, N}
        offset_in = blockDim().x * (blockIdx().x - 1)

        index_in = offset_in + threadIdx().x

        if index_in <= length(input)
            element = @inbounds input[index_in]

            # the index of an element in the original array along dimension that we will flip
            #assume(numelemsinprevdims > 0)
            #assume(numelemsincurrdim > 0)
            ik = ((cld(index_in, numelemsinprevdims) - 1) % numelemsincurrdim) + 1

            index_out = index_in + (numelemsincurrdim - 2ik + 1) * numelemsinprevdims

            @inbounds output[index_out] = element
        end

        return
    end

    nthreads = 256
    nblocks = cld(prod(shape), nthreads)
    shmem = nthreads * sizeof(T)

    @cuda threads=nthreads blocks=nblocks kernel(input, output)
end

# in-place version, swapping two elements on half the number of threads
function _reverse(data::CuArray{T, N}; dims::Integer=1) where {T, N}
    shape = [size(data)...]
    numelemsinprevdims = prod(shape[1:dims-1])
    numelemsincurrdim = shape[dims]

    function kernel(data::CuDeviceArray{T, N}) where {T, N}
        offset_in = blockDim().x * (blockIdx().x - 1)

        index_in = offset_in + threadIdx().x

        # the index of an element in the original array along dimension that we will flip
        #assume(numelemsinprevdims > 0)
        #assume(numelemsincurrdim > 0)
        ik = ((cld(index_in, numelemsinprevdims) - 1) % numelemsincurrdim) + 1

        index_out = index_in + (numelemsincurrdim - 2ik + 1) * numelemsinprevdims

        if index_in <= length(data) && index_in < index_out
            @inbounds begin
                temp = data[index_out]
                data[index_out] = data[index_in]
                data[index_in] = temp
            end
        end

        return
    end

    # NOTE: we launch twice the number of threads, which is wasteful, but the ND index
    #       calculations don't allow using only the first half of the threads
    #       (e.g. [1 2 3; 4 5 6] where threads 1 and 2 swap respectively (1,2) and (2,1)).

    nthreads = 256
    nblocks = cld(prod(shape), nthreads)
    shmem = nthreads * sizeof(T)

    @cuda threads=nthreads blocks=nblocks kernel(data)
end


# n-dimensional API

# in-place
function Base.reverse!(data::CuArray{T, N}; dims::Integer) where {T, N}
    if !(1 ≤ dims ≤ length(size(data)))
      ArgumentError("dimension $dims is not 1 ≤ $dims ≤ $length(size(input))")
    end

    _reverse(data; dims=dims)

    return data
end

# out-of-place
function Base.reverse(input::CuArray{T, N}; dims::Integer) where {T, N}
    if !(1 ≤ dims ≤ length(size(input)))
      ArgumentError("dimension $dims is not 1 ≤ $dims ≤ $length(size(input))")
    end

    output = similar(input)
    _reverse(input, output; dims=dims)

    return output
end


# 1-dimensional API

# in-place
function Base.reverse!(data::CuVector{T}, start=1, stop=length(data)) where {T}
    _reverse(view(data, start:stop))
    return data
end

# out-of-place
function Base.reverse(input::CuVector{T}, start=1, stop=length(input)) where {T}
    output = similar(input)

    start > 1 && copyto!(output, 1, input, 1, start-1)
    _reverse(view(input, start:stop), view(output, start:stop))
    stop < length(input) && copyto!(output, stop+1, input, stop+1)

    return output
end


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
