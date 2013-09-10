# Arrays on GPU

typealias CUdeviceptr Cuint

immutable GPtr
	p::CUdeviceptr

	GPtr() = new(convert(CUdeviceptr, 0))
	GPtr(p::CUdeviceptr) = new(p)
end

function free(p::GPtr)
	@cucall(:cuMemFree, (CUdeviceptr,), p.p)
end

isnull(p::GPtr) = (p.p == 0)


#################################################
#
#  GVector: 1D array on GPU
#
#################################################

type GVector{T}
	ptr::GPtr
	len::Int

	function GVector(len::Integer)
		a = CUdeviceptr[0]
		nbytes = int(len) * sizeof(T)
		@cucall(:cuMemAlloc, (Ptr{CUdeviceptr}, Csize_t), a, nbytes)
		p = GPtr(a[1])
		new(p, int(len))
	end
end

length(g::GVector) = g.len
size(g::GVector) = (g.len,)
size(g::GVector, d::Integer) = d == 1 ? g.len : d > 1 ? 1 : error("Invalid dim")

function free(g::GVector)
	if !isnull(g.ptr)
		free(g.ptr)
		g.ptr = GPtr()
	end
end

function copy!{T}(dst::Array{T}, src::GVector{T})
	if length(dst) != length(src)
		throw(ArgumentError("Inconsistent array length."))
	end
	nbytes = length(src) * sizeof(T)
	@cucall(:cuMemcpyDtoH, (Ptr{Void}, CUdeviceptr, Csize_t), pointer(dst), src.ptr.p, nbytes)
	return dst
end

function copy!{T}(dst::GVector{T}, src::Array{T})
	if length(dst) != length(src)
		throw(ArgumentError("Inconsistent array length."))
	end
	nbytes = length(src) * sizeof(T)
	@cucall(:cuMemcpyHtoD, (CUdeviceptr, Ptr{Void}, Csize_t), dst.ptr.p, pointer(src), nbytes)
	return dst
end

GVector{T}(a::Array{T}) = copy!(GVector{T}(length(a)), a)
to_gpu{T}(a::Vector{T}) = GVector{T}(a)
to_host{T}(g::GVector{T}) = copy!(Array(T, length(g)), g)


#################################################
#
#  GMatrix: 2D array on GPU
#
#################################################

type GMatrix{T}  # like Arrays in Julia, GMatrix is column-major
	ptr::GPtr
	nrows::Int  # number of rows (i.e. length of each column)
	ncols::Int  # number of columns
	pitch::Int  # number of bytes per column

	function GMatrix(m::Integer, n::Integer)
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

length(g::GMatrix) = g.nrows * g.ncols
size(g::GMatrix) = (g.nrows, g.ncols)

function size(g::GMatrix, dim::Integer)
	d == 1 ? g.nrows : 
	d == 2 ? g.ncols : 
	d > 2 ? 1 : error("Invalid dim")
end

function free(g::GMatrix)
	if !isnull(g.ptr)
		free(g.ptr)
		g.ptr = GPtr()
	end
end


