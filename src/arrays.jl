# Arrays on GPU

typealias CUdeviceptr Cuint

immutable CuPtr
	p::CUdeviceptr

	CuPtr() = new(convert(CUdeviceptr, 0))
	CuPtr(p::CUdeviceptr) = new(p)
end

function free(p::CuPtr)
	@cucall(:cuMemFree, (CUdeviceptr,), p.p)
end

isnull(p::CuPtr) = (p.p == 0)


#################################################
#
#  CuVector: 1D array on GPU
#
#################################################

type CuVector{T}
	ptr::CuPtr
	len::Int

	function CuVector(len::Integer)
		a = CUdeviceptr[0]
		nbytes = int(len) * sizeof(T)
		@cucall(:cuMemAlloc, (Ptr{CUdeviceptr}, Csize_t), a, nbytes)
		p = CuPtr(a[1])
		new(p, int(len))
	end
end

length(g::CuVector) = g.len
size(g::CuVector) = (g.len,)
size(g::CuVector, d::Integer) = d == 1 ? g.len : d > 1 ? 1 : error("Invalid dim")

function free(g::CuVector)
	if !isnull(g.ptr)
		free(g.ptr)
		g.ptr = CuPtr()
	end
end

function copy!{T}(dst::Array{T}, src::CuVector{T})
	if length(dst) != length(src)
		throw(ArgumentError("Inconsistent array length."))
	end
	nbytes = length(src) * sizeof(T)
	@cucall(:cuMemcpyDtoH, (Ptr{Void}, CUdeviceptr, Csize_t), pointer(dst), src.ptr.p, nbytes)
	return dst
end

function copy!{T}(dst::CuVector{T}, src::Array{T})
	if length(dst) != length(src)
		throw(ArgumentError("Inconsistent array length."))
	end
	nbytes = length(src) * sizeof(T)
	@cucall(:cuMemcpyHtoD, (CUdeviceptr, Ptr{Void}, Csize_t), dst.ptr.p, pointer(src), nbytes)
	return dst
end

CuVector{T}(a::Array{T}) = copy!(CuVector{T}(length(a)), a)
to_gpu{T}(a::Vector{T}) = CuVector{T}(a)
to_host{T}(g::CuVector{T}) = copy!(Array(T, length(g)), g)


#################################################
#
#  CuMatrix: 2D array on GPU
#
#################################################

type CuMatrix{T}  # like Arrays in Julia, CuMatrix is column-major
	ptr::CuPtr
	nrows::Int  # number of rows (i.e. length of each column)
	ncols::Int  # number of columns
	pitch::Int  # number of bytes per column

	function CuMatrix(m::Integer, n::Integer)
		aptr = CUdeviceptr[0]
		apitch = Csize_t[0]
		wbytes = int(m) * sizeof(T)
		@cucall(:cuMemAllocPitch, (Ptr{CUdeviceptr}, Ptr{Csize_t}, Csize_t, Csize_t, Cuint), 
			aptr, apitch, wbytes, n, 4)
		p = aptr[1]
		pitch = int(apitch[1])
		new(p, m, n, pitch)
	end
end

length(g::CuMatrix) = g.nrows * g.ncols
size(g::CuMatrix) = (g.nrows, g.ncols)

function size(g::CuMatrix, dim::Integer)
	d == 1 ? g.nrows : 
	d == 2 ? g.ncols : 
	d > 2 ? 1 : error("Invalid dim")
end

function free(g::CuMatrix)
	if !isnull(g.ptr)
		free(g.ptr)
		g.ptr = CuPtr()
	end
end


