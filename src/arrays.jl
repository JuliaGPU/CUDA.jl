# Arrays on GPU

typealias CUdeviceptr Cuint

immutable CuPtr
	p::CUdeviceptr

	CuPtr() = new(convert(CUdeviceptr, 0))
	CuPtr(p::CUdeviceptr) = new(p)
end

cubox(p::CuPtr) = cubox(p.p)

function cualloc(T::Type, len::Integer)
	a = CUdeviceptr[0]
	nbytes = int(len) * sizeof(T)
	@cucall(:cuMemAlloc, (Ptr{CUdeviceptr}, Csize_t), a, nbytes)
	return CuPtr(a[1])
end

function free(p::CuPtr)
	@cucall(:cuMemFree, (CUdeviceptr,), p.p)
end

isnull(p::CuPtr) = (p.p == 0)



#################################################
#
#  CuArray: contiguous array on GPU
#
#################################################

type CuArray{T,N}
	ptr::CuPtr
	shape::NTuple{N,Int}
	len::Int
end

function CuArray(T::Type, len::Integer)
	n = int(len)
	p = cualloc(T, n)
	CuArray{T,1}(p, (n,), n)
end

function CuArray{N}(T::Type, shape::NTuple{N,Int})
	n = prod(shape)
	p = cualloc(T, n)
	CuArray{T,N}(p, shape, n)
end

cubox(a::CuArray) = cubox(a.ptr)

length(g::CuArray) = g.len
size(g::CuArray) = g.shape
ndims{T,N}(g::CuArray{T,N}) = N
eltype{T,N}(g::CuArray{T,N}) = T

function size{T,N}(g::CuArray{T,N}, d::Integer) 
	d >= 1 ? (d <= N ? g.shape[d] : 1) : error("Invalid index of dimension.")
end

function free(g::CuArray)
	if !isnull(g.ptr)
		free(g.ptr)
		g.ptr = CuPtr()
	end
end

function copy!{T}(dst::Array{T}, src::CuArray{T})
	if length(dst) != length(src)
		throw(ArgumentError("Inconsistent array length."))
	end
	nbytes = length(src) * sizeof(T)
	@cucall(:cuMemcpyDtoH, (Ptr{Void}, CUdeviceptr, Csize_t), pointer(dst), src.ptr.p, nbytes)
	return dst
end

function copy!{T}(dst::CuArray{T}, src::Array{T})
	if length(dst) != length(src)
		throw(ArgumentError("Inconsistent array length."))
	end
	nbytes = length(src) * sizeof(T)
	@cucall(:cuMemcpyHtoD, (CUdeviceptr, Ptr{Void}, Csize_t), dst.ptr.p, pointer(src), nbytes)
	return dst
end

CuArray{T,N}(a::Array{T,N}) = copy!(CuArray(T, size(a)), a)
to_host{T}(g::CuArray{T}) = copy!(Array(T, size(g)), g)


