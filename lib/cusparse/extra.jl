export geam

"""
    geam(alpha::Number, A::CuSparseMatrix, beta::Number, B::CuSparseMatrix, index::SparseChar)

Performs `C = alpha * A + beta * B`. `A` and `B` are sparse matrices defined in CSR storage format.
"""
geam(alpha::Number, A::CuSparseMatrixCSR, beta::Number, B::CuSparseMatrixCSR, index::SparseChar)

"""
    geam(alpha::Number, A::CuSparseVector, beta::Number, B::CuSparseVector, index::SparseChar)

Performs `C = alpha * A + beta * B`. `A` and `B` are sparse vectors.
"""
geam(alpha::Number, A::CuSparseVector, beta::Number, B::CuSparseVector, index::SparseChar)

for (bname,gname,elty) in ((:cusparseScsrgeam2_bufferSizeExt, :cusparseScsrgeam2, :Float32),
                           (:cusparseDcsrgeam2_bufferSizeExt, :cusparseDcsrgeam2, :Float64),
                           (:cusparseCcsrgeam2_bufferSizeExt, :cusparseCcsrgeam2, :ComplexF32),
                           (:cusparseZcsrgeam2_bufferSizeExt, :cusparseZcsrgeam2, :ComplexF64))
    @eval begin
        function geam(alpha::Number, A::CuSparseMatrixCSR{$elty}, beta::Number, B::CuSparseMatrixCSR{$elty}, index::SparseChar)
            m, n = size(A)
            (m, n) == size(B) && DimensionMismatch("dimensions must match: a has dims $(axes(A)), b has dims $(axes(B))")
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
            return CuSparseMatrixCSR(rowPtrC, colValC, nzValC, (m, n))
        end

        function geam(alpha::Number, A::CuSparseVector{$elty}, beta::Number, B::CuSparseVector{$elty}, index::SparseChar)
            # A CuSparseVector is similar to a CuSparseMatrixCSR with one row.
            m = 1
            n = length(A)
            n == length(B) && DimensionMismatch("dimensions must match: a has length $(length(A)), b has length $(length(B))")
            descrA = CuMatrixDescriptor('G', 'L', 'N', index)
            descrB = CuMatrixDescriptor('G', 'L', 'N', index)
            descrC = CuMatrixDescriptor('G', 'L', 'N', index)

            rowPtrA = CuVector{Int32}([1; nnz(A)+1])
            rowPtrB = CuVector{Int32}([1; nnz(B)+1])
            rowPtrC = CuVector{Int32}(undef, m+1)
            local colValC, nzValC

            function bufferSize()
                out = Ref{Csize_t}()
                $bname(handle(), m, n,
                    alpha, descrA, nnz(A), nonzeros(A), rowPtrA, A.iPtr,
                    beta, descrB, nnz(B), nonzeros(B), rowPtrB, B.iPtr,
                    descrC, CU_NULL, rowPtrC, CU_NULL,
                    out)
                return out[]
            end

            with_workspace(bufferSize) do buffer
                nnzTotal = Ref{Cint}()
                cusparseXcsrgeam2Nnz(handle(), m, n,
                    descrA, nnz(A), rowPtrA, A.iPtr,
                    descrB, nnz(B), rowPtrB, B.iPtr,
                    descrC, rowPtrC, nnzTotal,
                    buffer)

                colValC = CuVector{Int32}(undef, nnzTotal[])
                nzValC  = CuVector{$elty}(undef, nnzTotal[])

                $gname(handle(), m, n,
                    alpha, descrA, nnz(A), nonzeros(A), rowPtrA, A.iPtr,
                    beta, descrB, nnz(B), nonzeros(B), rowPtrB, B.iPtr,
                    descrC, nzValC, rowPtrC, colValC,
                    buffer)
            end
            return CuSparseVector(colValC, nzValC, n)
        end
    end
end
