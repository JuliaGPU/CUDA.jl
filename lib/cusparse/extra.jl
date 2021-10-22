export geam

"""
    geam(alpha::Number, A::CuSparseMatrix, beta::Number, B::CuSparseMatrix, index::SparseChar)

Performs `C = alpha * A + beta * B`. `A` and `B` are sparse matrix defined in CSR storage format.
"""
geam(alpha::Number, A::CuSparseMatrixCSR, beta::Number, B::CuSparseMatrixCSR, index::SparseChar)

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

            rowPtrC = CuArray{Int32,1}(undef, m+1)

            function bufferSize()
                out = Ref{Csize_t}(1)
                $bname(handle(), m, n,
                    alpha, descrA, nnz(A), nonzeros(A), A.rowPtr, A.colVal,
                    beta, descrB, nnz(B), nonzeros(B), B.rowPtr, B.colVal,
                    descrC, CuArray{$elty,1}(undef, 0), rowPtrC, CuArray{Int32,1}(undef, 0),
                    out)
                return out[]
            end

            C = with_workspace(bufferSize) do buffer
                function get_nnzC(buffer)
                    nnzTotalDevHostPtr = Ref{Cint}(1)
                    cusparseXcsrgeam2Nnz(handle(), m, n,
                        descrA, nnz(A), A.rowPtr, A.colVal,
                        descrB, nnz(B), B.rowPtr, B.colVal,
                        descrC, rowPtrC, nnzTotalDevHostPtr,
                        buffer)
                    return nnzTotalDevHostPtr[]
                end

                nnzC = get_nnzC(buffer)
                colValC = CuArray{Int32,1}(undef, Int(nnzC))
                nzValC = CuArray{$elty,1}(undef, Int(nnzC))

                $gname(handle(), m, n,
                    alpha, descrA, nnz(A), nonzeros(A), A.rowPtr, A.colVal,
                    beta, descrB, nnz(B), nonzeros(B), B.rowPtr, B.colVal,
                    descrC, nzValC, rowPtrC, colValC,
                    buffer)
                return CuSparseMatrixCSR(rowPtrC, colValC, nzValC, (m, n))
            end
            C
        end
    end
end
