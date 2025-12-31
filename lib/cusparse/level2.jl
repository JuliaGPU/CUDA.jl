# sparse linear algebra functions that perform operations between sparse matrices and dense
# vectors

export sv2!, sv2, gemvi!

for (fname,elty) in ((:cusparseSbsrmv, :Float32),
                     (:cusparseDbsrmv, :Float64),
                     (:cusparseCbsrmv, :ComplexF32),
                     (:cusparseZbsrmv, :ComplexF64))
    @eval begin
        function mv!(transa::SparseChar,
                     alpha::Number,
                     A::CuSparseMatrixBSR{$elty},
                     X::CuVector{$elty},
                     beta::Number,
                     Y::CuVector{$elty},
                     index::SparseChar)

            # Support transa = 'C' for real matrices
            transa = $elty <: Real && transa == 'C' ? 'T' : transa

            desc = CuMatrixDescriptor('G', 'L', 'N', index)
            m,n = size(A)
            mb = div(m,A.blockDim)
            nb = div(n,A.blockDim)
            if transa == 'N'
                chkmvdims(X,n,Y,m)
            end
            if transa == 'T' || transa == 'C'
                chkmvdims(X,m,Y,n)
            end
            $fname(handle(), A.dir, transa, mb, nb,
                   nnz(A), alpha, desc, nonzeros(A), A.rowPtr,
                   A.colVal, A.blockDim, X, beta, Y)
            Y
        end
    end
end

"""
    sv2!(transa::SparseChar, uplo::SparseChar, diag::SparseChar, alpha::BlasFloat, A::CuSparseMatrixBSR, X::CuVector, index::SparseChar)

Performs `X = alpha * op(A) \\ X`, where `op` can be nothing (`transa = N`), tranpose
(`transa = T`) or conjugate transpose (`transa = C`). `X` is a dense vector, and `uplo`
tells `sv2!` which triangle of the block sparse matrix `A` to reference.
If the triangle has unit diagonal, set `diag` to 'U'.
"""
sv2!(transa::SparseChar, uplo::SparseChar, diag::SparseChar, alpha::BlasFloat, A::CuSparseMatrixBSR, X::CuVector, index::SparseChar)

# bsrsv2
for (bname,aname,sname,elty) in ((:cusparseSbsrsv2_bufferSize, :cusparseSbsrsv2_analysis, :cusparseSbsrsv2_solve, :Float32),
                                 (:cusparseDbsrsv2_bufferSize, :cusparseDbsrsv2_analysis, :cusparseDbsrsv2_solve, :Float64),
                                 (:cusparseCbsrsv2_bufferSize, :cusparseCbsrsv2_analysis, :cusparseCbsrsv2_solve, :ComplexF32),
                                 (:cusparseZbsrsv2_bufferSize, :cusparseZbsrsv2_analysis, :cusparseZbsrsv2_solve, :ComplexF64))
    @eval begin
        function sv2!(transa::SparseChar,
                      uplo::SparseChar,
                      diag::SparseChar,
                      alpha::Number,
                      A::CuSparseMatrixBSR{$elty},
                      X::CuVector{$elty},
                      index::SparseChar)

            # Support transa = 'C' for real matrices
            transa = $elty <: Real && transa == 'C' ? 'T' : transa

            desc = CuMatrixDescriptor('G', uplo, diag, index)
            m,n = size(A)
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mb = cld(m, A.blockDim)
            mX = length(X)
            if mX != m
                throw(DimensionMismatch("X must have length $m, but has length $mX"))
            end
            info = bsrsv2Info_t[0]
            cusparseCreateBsrsv2Info(info)

            function bufferSize()
                out = Ref{Cint}(1)
                $bname(handle(), A.dir, transa, mb, A.nnzb,
                       desc, nonzeros(A), A.rowPtr, A.colVal, A.blockDim,
                       info[1], out)
                return out[]
            end
            with_workspace(bufferSize) do buffer
                $aname(handle(), A.dir, transa, mb, A.nnzb,
                        desc, nonzeros(A), A.rowPtr, A.colVal, A.blockDim,
                        info[1], CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                posit = Ref{Cint}(1)
                cusparseXbsrsv2_zeroPivot(handle(), info[1], posit)
                if posit[] >= 0
                    error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                end
                $sname(handle(), A.dir, transa, mb, A.nnzb,
                        alpha, desc, nonzeros(A), A.rowPtr, A.colVal,
                        A.blockDim, info[1], X, X,
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            end
            cusparseDestroyBsrsv2Info(info[1])
            X
        end
    end
end

function sv2(transa::SparseChar, uplo::SparseChar, diag::SparseChar, alpha::Number,
             A::CuSparseMatrixBSR{T}, X::CuVector{T}, index::SparseChar) where T
    sv2!(transa,uplo,diag,alpha,A,copy(X),index)
end
function sv2(transa::SparseChar, uplo::SparseChar, diag::SparseChar,
             A::CuSparseMatrixBSR{T}, X::CuVector{T}, index::SparseChar) where T
    sv2!(transa,uplo,diag,one(T),A,copy(X),index)
end
function sv2(transa::SparseChar, uplo::SparseChar,
             A::CuSparseMatrixBSR{T}, X::CuVector{T}, index::SparseChar) where T
    sv2!(transa,uplo,'N',one(T),A,copy(X),index)
end

# gemvi!
for (bname,fname,elty) in ((:cusparseSgemvi_bufferSize, :cusparseSgemvi, :Float32),
                           (:cusparseDgemvi_bufferSize, :cusparseDgemvi, :Float64),
                           (:cusparseCgemvi_bufferSize, :cusparseCgemvi, :ComplexF32),
                           (:cusparseZgemvi_bufferSize, :cusparseZgemvi, :ComplexF64))
    @eval begin
        function gemvi!(transa::SparseChar,
                        alpha::Number,
                        A::CuMatrix{$elty},
                        X::CuSparseVector{$elty},
                        beta::Number,
                        Y::CuVector{$elty},
                        index::SparseChar)

            # Support transa = 'C' for real matrices
            transa = $elty <: Real && transa == 'C' ? 'T' : transa
            $elty <: Complex && transa == 'C' && throw(ArgumentError("Dense matrix - sparse vector multiplication with the adjoint of a complex matrix is not supported."))

            m,n = size(A)
            lda = max(1,stride(A,2))
            nnzX = nnz(X)

            function bufferSize()
                out = Ref{Cint}()
                $bname(handle(), transa, m, n, nnzX, out)
                return out[]
            end
            with_workspace(bufferSize) do buffer
                $fname(handle(), transa, m, n, alpha, A, lda, nnzX, nonzeros(X), nonzeroinds(X), beta, Y, index, buffer)
            end
            Y
        end
    end
end
