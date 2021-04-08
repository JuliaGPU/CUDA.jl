import Base.kron

export kron, kron!

function kron!(A::CuDeviceArray{<:Number},B::CuDeviceArray{<:Number},C::CuDeviceArray{<:Number})

    C_rows = size(C,1)
    B_rows = size(B,1)
    B_cols = size(B,2)

    index1 = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride1 = blockDim().x * gridDim().x

    index2 = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride2 = blockDim().y * gridDim().y

    index3 = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    stride3 = blockDim().z * gridDim().z

    @inbounds for j = index1:stride1:size(A,2), l = index2:stride2:B_cols, i = index3:stride3:size(A,1)
        aij = A[i,j]
        @inbounds for k = 1:B_rows
            C[(B_rows*(i-1) + k) + C_rows*((B_cols*(j-1) + l)-1)] = aij*B[k,l]
        end
    end
end

function kron(A::CuArray{S},B::CuArray{T}) where {S,T <: Number}

	col = size(A,2)*size(B,2)
	if col == one(col)
		C = CUDA.zeros(promote_type(S,T),size(A,1)*size(B,1))
	else
		C = CUDA.zeros(promote_type(S,T),size(A,1)*size(B,1),col)
	end

    nthreads = (8,8,8)
    nelements = size(A,1)*size(B,1)*col
    nblocks = cld(nelements, prod(nthreads))

    @cuda threads=nthreads blocks=nblocks kron!(A,B,C)
	return C
end
