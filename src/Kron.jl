import Base.kron

export kron, kron!

function kron!(A::CUDA.CuDeviceArray{<:Number},B::CUDA.CuDeviceArray{<:Number},C::CUDA.CuDeviceArray{<:Number})

    C_rows = size(C,1)
    B_rows = size(B,1)
    B_cols = size(B,2)

    index1 = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride1 = CUDA.blockDim().x * CUDA.gridDim().x

    index2 = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y
    stride2 = CUDA.blockDim().y * CUDA.gridDim().y

    index3 = (CUDA.blockIdx().z - 1) * CUDA.blockDim().z + CUDA.threadIdx().z
    stride3 = CUDA.blockDim().z * CUDA.gridDim().z

    @inbounds for j = index1:stride1:size(A,2), l = index2:stride2:B_cols, i = index3:stride3:size(A,1)
        aij = A[i,j]
        @inbounds for k = 1:B_rows
            C[(B_rows*(i-1) + k) + C_rows*((B_cols*(j-1) + l)-1)] = aij*B[k,l]
        end
    end
end

function kron(A::CUDA.CuArray{S},B::CUDA.CuArray{T}) where {S,T <: Number}
	# Need to specify thread count.

	# The if statement is to handle a nuance of CUDA where if a column count
	# is specified in CUDA.zeros, it will ALWAYS return a CuArray of dim=2.
	# Otherwise, this function would be incompatible with ket.
	col = size(A,2)*size(B,2)
	if col == one(col)
		C = CUDA.zeros(promote_type(S,T),size(A,1)*size(B,1))
	else
		C = CUDA.zeros(promote_type(S,T),size(A,1)*size(B,1),col)
	end
	CUDA.@cuda kron!(A,B,C)
	return C
end
