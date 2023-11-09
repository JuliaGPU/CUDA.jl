# sparse linear algebra functions that perform operations between sparse and (usually tall)
# dense matrices

export mm!, sm2!, sm2

"""
    mm!(transa::SparseChar, transb::SparseChar, alpha::Number, A::CuSparseMatrix, B::CuMatrix, beta::Number, C::CuMatrix, index::SparseChar)

Performs `C = alpha * op(A) * op(B) + beta * C`, where `op` can be nothing (`transa = N`),
tranpose (`transa = T`) or conjugate transpose (`transa = C`).
`B` and `C` are dense matrices.
"""
mm!(transa::SparseChar, transb::SparseChar, alpha::Number, A::CuSparseMatrix, B::CuMatrix, beta::Number, C::CuMatrix, index::SparseChar)

# bsrmm
for (fname,elty) in ((:cusparseSbsrmm, :Float32),
                     (:cusparseDbsrmm, :Float64),
                     (:cusparseCbsrmm, :ComplexF32),
                     (:cusparseZbsrmm, :ComplexF64))
    @eval begin
        function mm!(transa::SparseChar,
                     transb::SparseChar,
                     alpha::Number,
                     A::CuSparseMatrixBSR{$elty},
                     B::StridedCuMatrix{$elty},
                     beta::Number,
                     C::StridedCuMatrix{$elty},
                     index::SparseChar)

            # Support transa = 'C' and `transb = 'C' for real matrices
            transa = $elty <: Real && transa == 'C' ? 'T' : transa
            transb = $elty <: Real && transb == 'C' ? 'T' : transb

            desc = CuMatrixDescriptor('G', 'L', 'N', index)
            m,k = size(A)
            mb = cld(m, A.blockDim)
            kb = cld(k, A.blockDim)
            n = size(C)[2]
            if transa == 'N' && transb == 'N'
                chkmmdims(B,C,k,n,m,n)
            elseif transa == 'N' && transb != 'N'
                chkmmdims(B,C,n,k,m,n)
            elseif transa != 'N' && transb == 'N'
                chkmmdims(B,C,m,n,k,n)
            elseif transa != 'N' && transb != 'N'
                chkmmdims(B,C,n,m,k,n)
            end
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            $fname(handle(), A.dir,
                   transa, transb, mb, n, kb, A.nnzb,
                   alpha, desc, nonzeros(A),A.rowPtr, A.colVal,
                   A.blockDim, B, ldb, beta, C, ldc)
            C
        end
    end
end

"""
    sm2!(transa::SparseChar, transxy::SparseChar, uplo::SparseChar, diag::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, X::CuMatrix, index::SparseChar)

Performs `X = alpha * op(A) \\ op(X)`, where `op` can be nothing (`transa = N`), tranpose
(`transa = T`) or conjugate transpose (`transa = C`). `X` is a dense matrix, and `uplo`
tells `sm2!` which triangle of the block sparse matrix `A` to reference.
If the triangle has unit diagonal, set `diag` to 'U'.
"""
sm2!(transa::SparseChar, transxy::SparseChar, diag::SparseChar, alpha::Number, A::CuSparseMatrix, X::CuMatrix, index::SparseChar)

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
                throw(DimensionMismatch(""))
            end
            if transxy != 'N' && (nX != m)
                throw(DimensionMismatch(""))
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

# csrsm2
for (bname,aname,sname,elty) in ((:cusparseScsrsm2_bufferSizeExt, :cusparseScsrsm2_analysis, :cusparseScsrsm2_solve, :Float32),
                                 (:cusparseDcsrsm2_bufferSizeExt, :cusparseDcsrsm2_analysis, :cusparseDcsrsm2_solve, :Float64),
                                 (:cusparseCcsrsm2_bufferSizeExt, :cusparseCcsrsm2_analysis, :cusparseCcsrsm2_solve, :ComplexF32),
                                 (:cusparseZcsrsm2_bufferSizeExt, :cusparseZcsrsm2_analysis, :cusparseZcsrsm2_solve, :ComplexF64))
    @eval begin
        function sm2!(transa::SparseChar,
                      transxy::SparseChar,
                      uplo::SparseChar,
                      diag::SparseChar,
                      alpha::Number,
                      A::CuSparseMatrixCSR{$elty},
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
            mX,nX = size(X)
            nrhs = transxy == 'N' ? nX : mX
            if transxy == 'N' && (mX != m)
                throw(DimensionMismatch(""))
            end
            if transxy != 'N' && (nX != m)
                throw(DimensionMismatch(""))
            end
            ldx = max(1,stride(X,2))
            info = csrsm2Info_t[0]
            cusparseCreateCsrsm2Info(info)
            # use non block algo (0) for now...

            function bufferSize()
                out = Ref{UInt64}(1)
                $bname(handle(), 0, transa, transxy,
                        m, nrhs, nnz(A), alpha, desc, nonzeros(A), A.rowPtr,
                        A.colVal, X, ldx, info[], CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                        out)
                return out[]
            end
            with_workspace(bufferSize) do buffer
                $aname(handle(), 0, transa, transxy,
                        m, nrhs, nnz(A), alpha, desc, nonzeros(A), A.rowPtr,
                        A.colVal, X, ldx, info[],
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                posit = Ref{Cint}(1)
                cusparseXcsrsm2_zeroPivot(handle(), info[1], posit)
                if posit[] >= 0
                    error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                end
                $sname(handle(), 0, transa, transxy, m,
                        nrhs, nnz(A), alpha, desc, nonzeros(A), A.rowPtr,
                        A.colVal, X, ldx, info[],
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            end
            cusparseDestroyCsrsm2Info(info[])
            X
        end
    end
end

# cscsm2
for (bname,aname,sname,elty) in ((:cusparseScsrsm2_bufferSizeExt, :cusparseScsrsm2_analysis, :cusparseScsrsm2_solve, :Float32),
                                 (:cusparseDcsrsm2_bufferSizeExt, :cusparseDcsrsm2_analysis, :cusparseDcsrsm2_solve, :Float64),
                                 (:cusparseCcsrsm2_bufferSizeExt, :cusparseCcsrsm2_analysis, :cusparseCcsrsm2_solve, :ComplexF32),
                                 (:cusparseZcsrsm2_bufferSizeExt, :cusparseZcsrsm2_analysis, :cusparseZcsrsm2_solve, :ComplexF64))
    @eval begin
        function sm2!(transa::SparseChar,
                      transxy::SparseChar,
                      uplo::SparseChar,
                      diag::SparseChar,
                      alpha::Number,
                      A::CuSparseMatrixCSC{$elty},
                      X::StridedCuMatrix{$elty},
                      index::SparseChar)

            # Support transa = 'C' and transxy = 'C' for real matrices
            transa = $elty <: Real && transa == 'C' ? 'T' : transa
            transxy = $elty <: Real && transxy == 'C' ? 'T' : transxy

            ctransa = 'N'
            cuplo = 'U'
            if transa == 'N'
                ctransa = 'T'
            elseif transa == 'C' && $elty <: Complex
                throw(ArgumentError("Backward and forward sweeps with the adjoint of a complex CSC matrix is not supported. Use a CSR matrix instead."))
            end
            if uplo == 'U'
                cuplo = 'L'
            end
            desc = CuMatrixDescriptor('G', cuplo, diag, index)
            n,m = size(A)
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($n,$m)!"))
            end
            mX,nX = size(X)
            nrhs = transxy == 'N' ? nX : mX
            if transxy == 'N' && (mX != m)
                throw(DimensionMismatch(""))
            end
            if transxy != 'N' && (nX != m)
                throw(DimensionMismatch(""))
            end
            ldx = max(1,stride(X,2))
            info = csrsm2Info_t[0]
            cusparseCreateCsrsm2Info(info)
            # use non block algo (0) for now...

            function bufferSize()
                out = Ref{UInt64}(1)
                $bname(handle(), 0, ctransa, transxy,
                       m, nrhs, nnz(A), alpha, desc, nonzeros(A), A.colPtr,
                       rowvals(A), X, ldx, info[], CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                       out)
                return out[]
            end
            with_workspace(bufferSize) do buffer
                $aname(handle(), 0, ctransa, transxy,
                        m, nrhs, nnz(A), alpha, desc, nonzeros(A), A.colPtr,
                        rowvals(A), X, ldx, info[],
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                posit = Ref{Cint}(1)
                cusparseXcsrsm2_zeroPivot(handle(), info[1], posit)
                if posit[] >= 0
                    error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                end
                $sname(handle(), 0, ctransa, transxy, m,
                        nrhs, nnz(A), alpha, desc, nonzeros(A), A.colPtr,
                        rowvals(A), X, ldx, info[],
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            end
            cusparseDestroyCsrsm2Info(info[])
            X
        end
    end
end
function sm2(transa::SparseChar, transxy::SparseChar, uplo::SparseChar, diag::SparseChar,
             alpha::Number, A::CuSparseMatrix{T}, X::StridedCuMatrix{T},
             index::SparseChar) where T
    sm2!(transa,transxy,uplo,diag,alpha,A,copy(X),index)
end
function sm2(transa::SparseChar, transxy::SparseChar, uplo::SparseChar, diag::SparseChar,
              A::CuSparseMatrix{T}, X::StridedCuMatrix{T}, index::SparseChar) where T
    sm2!(transa,transxy,uplo,diag,one(T),A,copy(X),index)
end
function sm2(transa::SparseChar, transxy::SparseChar, uplo::SparseChar,
             A::CuSparseMatrix{T}, X::StridedCuMatrix{T}, index::SparseChar) where T
    sm2!(transa,transxy,uplo,'N',one(T),A,copy(X),index)
end
