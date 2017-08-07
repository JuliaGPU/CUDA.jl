const _allowslow = Ref(true)

allowslow(flag = true) = (_allowslow[] = flag)

function assertslow(op = "Operation")
  _allowslow[] || error("$op is disabled")
  return
end

Base.IndexStyle(::Type{<:CuArray}) = IndexLinear()

function _getindex(xs::CuArray{T}, i::Integer) where T
  x = Array{T}(1)
  ptr = OwnedPtr{T}(xs.ptr.ptr + (i-1)*sizeof(T), xs.ptr.ctx)
  Mem.download(pointer(x), ptr, sizeof(T))
  return x[1]
end

function Base.getindex{T}(xs::CuArray{T}, i::Integer)
  assertslow("getindex")
  _getindex(xs, i)
end

function Base.setindex!{T}(xs::CuArray{T}, v::T, i::Integer)
  assertslow("setindex!")
  x = T[v]
  ptr = OwnedPtr{T}(xs.ptr.ptr + (i-1)*sizeof(T), xs.ptr.ctx)
  Mem.upload(ptr, pointer(x), sizeof(T))
  return x[1]
end

Base.setindex!(xs::CuArray, v, i::Integer) = xs[i] = convert(eltype(xs), v)

using Base.Cartesian

@generated function index_kernel(dest::AbstractArray, src::AbstractArray, idims, Is)
    N = length(Is.parameters)
    quote
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        is = ind2sub(idims, i)
        @nexprs $N i -> @inbounds I_i = Is[i][is[i]]
        @inbounds dest[i] = @ncall $N getindex src i -> I_i
        return
    end
end

function Base._unsafe_getindex!(dest::CuArray, src::CuArray, Is::Union{Real, AbstractArray}...)
    idims = map(length, Is)
    blk, thr = cudims(length(dest))
    @cuda (blk, thr) index_kernel(dest, src, idims, Is)
    return dest
end
