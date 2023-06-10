using CUDA, CUDA.CUSPARSE

using LinearAlgebra
using SparseArrays

m = 15
n = 25
k = 35
b = 2
p = 0.5

elty = Float32

α = rand(elty)
β = rand(elty)

# C = αAB + βC
A1 = CuSparseMatrixCSR{elty}(sprandn(elty, m, k, p))
A2 = copy(A1)
A2.nzVal = CUDA.randn(size(A2.nzVal)...)
A = CUSPARSE.batchcat(A1, A2)

B = CUDA.randn(elty, k, n, b)
C = CUDA.randn(elty, m, n, b)
D = copy(C)

CUSPARSE.bmm!('N', 'N', α, A, B, β, C, 'O') 

D[:,:,1] = α * A1 * B[:,:,1] + β * D[:,:,1]
D[:,:,2] = α * A2 * B[:,:,2] + β * D[:,:,2]

@show D ≈ C



# C = αAᵀB + βC
A1 = CuSparseMatrixCSR{elty}(sprandn(elty, k, m, p))
A2 = copy(A1)
A2.nzVal = CUDA.randn(size(A2.nzVal)...)
A = CUSPARSE.batchcat(A1, A2)

B = CUDA.randn(elty, k, n, b)
C = CUDA.randn(elty, m, n, b)
D = copy(C)

CUSPARSE.bmm!('T', 'N', α, A, B, β, C, 'O') 

D[:,:,1] = α * A1' * B[:,:,1] + β * D[:,:,1]
D[:,:,2] = α * A2' * B[:,:,2] + β * D[:,:,2]

@show D ≈ C


# C = αABᵀ + βC
A1 = CuSparseMatrixCSR{elty}(sprandn(elty, m, k, p))
A2 = copy(A1)
A2.nzVal = CUDA.randn(size(A2.nzVal)...)
A = CUSPARSE.batchcat(A1, A2)

B = CUDA.randn(elty, n, k, b)
C = CUDA.randn(elty, m, n, b)
D = copy(C)

CUSPARSE.bmm!('N', 'T', α, A, B, β, C, 'O') 

D[:,:,1] = α * A1 * B[:,:,1]' + β * D[:,:,1]
D[:,:,2] = α * A2 * B[:,:,2]' + β * D[:,:,2]

@show D ≈ C


# C = αAᵀBᵀ + βC
A1 = CuSparseMatrixCSR{elty}(sprandn(elty, k, m, p))
A2 = copy(A1)
A2.nzVal = CUDA.randn(size(A2.nzVal)...)
A = CUSPARSE.batchcat(A1, A2)

B = CUDA.randn(elty, n, k, b)
C = CUDA.randn(elty, m, n, b)
D = copy(C)

CUSPARSE.bmm!('T', 'T', α, A, B, β, C, 'O') 

D[:,:,1] = α * A1' * B[:,:,1]' + β * D[:,:,1]
D[:,:,2] = α * A2' * B[:,:,2]' + β * D[:,:,2]

@show D ≈ C
