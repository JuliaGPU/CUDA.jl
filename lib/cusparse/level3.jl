# sparse linear algebra functions that perform operations between sparse and (usually tall)
# dense matrices

export sm2!, sm2

"""
    sm2!(transa::SparseChar, transxy::SparseChar, uplo::SparseChar, diag::SparseChar, alpha::BlasFloat, A::CuSparseMatrixBSR, X::CuMatrix, index::SparseChar)

Performs `X = alpha * op(A) \\ op(X)`, where `op` can be nothing (`transa = N`), tranpose
(`transa = T`) or conjugate transpose (`transa = C`). `X` is a dense matrix, and `uplo`
tells `sm2!` which triangle of the block sparse matrix `A` to reference.
If the triangle has unit diagonal, set `diag` to 'U'.
"""
sm2!(transa::SparseChar, transxy::SparseChar, diag::SparseChar, alpha::Number, A::CuSparseMatrixBSR, X::CuMatrix, index::SparseChar)

# bsrsm2
for (bname,aname,sname,elty) in ((:cusparseSbsrsm2_bufferSize, :cusparseSbsrsm2_analysis, :cusparseSbsrsm2_solve, :Float32),
                                 (:cusparseDbsrsm2_bufferSize, :cusparseDbsrsm2_analysis, :cusparseDbsrsm2_solve, :Float64),
                                 (:cusparseCbsrsm2_bufferSize, :cusparseCbsrsm2_analysis, :cusparseCbsrsm2_solve, :ComplexF32),
                                 (:cusparseZbsrsm2_bufferSize, :cusparseZbsrsm2_analysis, :cusparseZbsrsm2_solve, :ComplexF64))
    @eval begin
        function sm2!(transa::SparseChar,
                      transxy::SparseChar,
                      uplo::SparseChar,
                      diag::SparseChar,
                      alpha::Number,
                      A::CuSparseMatrixBSR{$elty},
                      X::StridedCuMatrix{$elty},
                      index::SparseChar)

            # Support transa = 'C' and transxy = 'C' for real matrices
            transa = $elty <: Real && transa == 'C' ? 'T' : transa
            transxy = $elty <: Real && transxy == 'C' ? 'T' : transxy

            desc = CuMatrixDescriptor('G', uplo, diag, index)
            m,n = size(A)
            if m != n
                 throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mb = cld(m, A.blockDim)
            mX,nX = size(X)
            nrhs = transxy == 'N' ? nX : mX
            if transxy == 'N' && (mX != m)
                throw(DimensionMismatch("first dimensions of A ($m) and X ($mX) must match when transxy is 'N'"))
            end
            if transxy != 'N' && (nX != m)
                throw(DimensionMismatch("first dimension of A ($m) must match second dimension of X ($nX) when transxy is not 'N'"))
            end
            ldx = max(1,stride(X,2))
            info = bsrsm2Info_t[0]
            cusparseCreateBsrsm2Info(info)

            function bufferSize()
                out = Ref{Cint}(1)
                $bname(handle(), A.dir, transa, transxy,
                       mb, nrhs, A.nnzb, desc, nonzeros(A), A.rowPtr,
                       A.colVal, A.blockDim, info[],
                       out)
                return out[]
            end
            with_workspace(bufferSize) do buffer
                $aname(handle(), A.dir, transa, transxy,
                        mb, nrhs, A.nnzb, desc, nonzeros(A), A.rowPtr,
                        A.colVal, A.blockDim, info[],
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                posit = Ref{Cint}(1)
                cusparseXbsrsm2_zeroPivot(handle(), info[], posit)
                if posit[] >= 0
                    error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                end
                $sname(handle(), A.dir, transa, transxy, mb,
                        nrhs, A.nnzb, alpha, desc, nonzeros(A), A.rowPtr,
                        A.colVal, A.blockDim, info[], X, ldx, X, ldx,
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            end
            cusparseDestroyBsrsm2Info(info[1])
            X
        end
    end
end

function sm2(transa::SparseChar, transxy::SparseChar, uplo::SparseChar, diag::SparseChar,
             alpha::Number, A::CuSparseMatrixBSR{T}, X::StridedCuMatrix{T},
             index::SparseChar) where T
    sm2!(transa,transxy,uplo,diag,alpha,A,copy(X),index)
end
function sm2(transa::SparseChar, transxy::SparseChar, uplo::SparseChar, diag::SparseChar,
              A::CuSparseMatrixBSR{T}, X::StridedCuMatrix{T}, index::SparseChar) where T
    sm2!(transa,transxy,uplo,diag,one(T),A,copy(X),index)
end
function sm2(transa::SparseChar, transxy::SparseChar, uplo::SparseChar,
             A::CuSparseMatrixBSR{T}, X::StridedCuMatrix{T}, index::SparseChar) where T
    sm2!(transa,transxy,uplo,'N',one(T),A,copy(X),index)
end
