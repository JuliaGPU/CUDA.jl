export geam, axpby

"""
    geam(alpha::Number, A::CuSparseMatrix, beta::Number, B::CuSparseMatrix, index::SparseChar)

Performs `C = alpha * A + beta * B`. `A` and `B` are sparse matrices defined in CSR storage format.
"""
geam(alpha::Number, A::CuSparseMatrixCSR, beta::Number, B::CuSparseMatrixCSR, index::SparseChar)

for (bname,gname,elty) in ((:cusparseScsrgeam2_bufferSizeExt, :cusparseScsrgeam2, :Float32),
                           (:cusparseDcsrgeam2_bufferSizeExt, :cusparseDcsrgeam2, :Float64),
                           (:cusparseCcsrgeam2_bufferSizeExt, :cusparseCcsrgeam2, :ComplexF32),
                           (:cusparseZcsrgeam2_bufferSizeExt, :cusparseZcsrgeam2, :ComplexF64))
    @eval begin
        function geam(alpha::Number, A::CuSparseMatrixCSR{$elty}, beta::Number, B::CuSparseMatrixCSR{$elty}, index::SparseChar)
            m, n = size(A)
            (m, n) == size(B) || throw(DimensionMismatch("dimensions must match: A has dims $(size(A)), B has dims $(size(B))"))
            descrA = CuMatrixDescriptor('G', 'L', 'N', index)
            descrB = CuMatrixDescriptor('G', 'L', 'N', index)
            descrC = CuMatrixDescriptor('G', 'L', 'N', index)

            rowPtrC = CuVector{Int32}(undef, m+1)
            local colValC, nzValC

            function bufferSize()
                out = Ref{Csize_t}()
                $bname(handle(), m, n,
                    alpha, descrA, nnz(A), nonzeros(A), A.rowPtr, A.colVal,
                    beta, descrB, nnz(B), nonzeros(B), B.rowPtr, B.colVal,
                    descrC, CU_NULL, rowPtrC, CU_NULL,
                    out)
                return out[]
            end

            with_workspace(bufferSize) do buffer
                nnzTotal = Ref{Cint}()
                cusparseXcsrgeam2Nnz(handle(), m, n,
                    descrA, nnz(A), A.rowPtr, A.colVal,
                    descrB, nnz(B), B.rowPtr, B.colVal,
                    descrC, rowPtrC, nnzTotal,
                    buffer)

                colValC = CuVector{Int32}(undef, nnzTotal[])
                nzValC  = CuVector{$elty}(undef, nnzTotal[])

                $gname(handle(), m, n,
                    alpha, descrA, nnz(A), nonzeros(A), A.rowPtr, A.colVal,
                    beta, descrB, nnz(B), nonzeros(B), B.rowPtr, B.colVal,
                    descrC, nzValC, rowPtrC, colValC,
                    buffer)
            end

            C = CuSparseMatrixCSR(rowPtrC, colValC, nzValC, (m, n))
            return C
        end
    end
end

"""
    axpby(alpha::Number, x::CuSparseVector, beta::Number, y::CuSparseVector, index::SparseChar)

Performs `z = alpha * x + beta * y`. `x` and `y` are sparse vectors.
"""
axpby(alpha::Number, x::CuSparseVector, beta::Number, y::CuSparseVector, index::SparseChar)

function axpby(alpha::Number, x::CuSparseVector{T}, beta::Number, y::CuSparseVector{T}, index::SparseChar) where {T <: BlasFloat}
    n = length(x)
    n == length(y) || throw(DimensionMismatch("dimensions must match: x has length $(length(x)), y has length $(length(y))"))

    # we model x as a CuSparseMatrixCSR with one row.
    rowPtrA = CuVector{Int32}([1; nnz(x)+1])
    A = CuSparseMatrixCSR(rowPtrA, x.iPtr, nonzeros(x), (1,n))

    # we model y as a CuSparseMatrixCSR with one row.
    rowPtrB = CuVector{Int32}([1; nnz(y)+1])
    B = CuSparseMatrixCSR(rowPtrB, y.iPtr, nonzeros(y), (1,n))

    C = geam(alpha, A, beta, B, index)
    z = CuSparseVector(C.colVal, C.nzVal, n)
    return z
end
