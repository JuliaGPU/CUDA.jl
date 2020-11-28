# sparse linear algebra functions that perform operations between sparse and (usually tall)
# dense matrices

export mm!, mm2!, sm2!, sm2

"""
    mm!(transa::SparseChar, transb::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, B::CuMatrix, beta::BlasFloat, C::CuMatrix, index::SparseChar)

Performs `C = alpha * op(A) * op(B) + beta * C`, where `op` can be nothing (`transa = N`),
tranpose (`transa = T`) or conjugate transpose (`transa = C`).
`A` is a sparse matrix defined in BSR storage format. `B` and `C` are dense matrices.
"""
mm!(transa::SparseChar, transb::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, B::CuMatrix, beta::BlasFloat, C::CuMatrix, index::SparseChar)

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
                     B::CuMatrix{$elty},
                     beta::Number,
                     C::CuMatrix{$elty},
                     index::SparseChar)
            desc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, index)
            m,k = A.dims
            mb = div(m,A.blockDim)
            kb = div(k,A.blockDim)
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
                   transa, transb, mb, n, kb, nnz(A),
                   alpha, desc, nonzeros(A),A.rowPtr, A.colVal,
                   A.blockDim, B, ldb, beta, C, ldc)
            C
        end
    end
end

for (fname,elty) in ((:cusparseScsrmm2, :Float32),
                     (:cusparseDcsrmm2, :Float64),
                     (:cusparseCcsrmm2, :ComplexF32),
                     (:cusparseZcsrmm2, :ComplexF64))
    @eval begin
        function mm2!(transa::SparseChar,
                      transb::SparseChar,
                      alpha::Number,
                      A::CuSparseMatrixCSR{$elty},
                      B::CuMatrix{$elty},
                      beta::Number,
                      C::CuMatrix{$elty},
                      index::SparseChar)
            if transb == 'C'
                throw(ArgumentError("B^H is not supported"))
            elseif transb == 'T' && transa != 'N'
                throw(ArgumentError("When using B^T, A can be neither transposed nor adjointed"))
            end
            desc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, index)
            m,k = A.dims
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
            $fname(handle(),
                   transa, transb, m, n, k, nnz(A), alpha, desc,
                   nonzeros(A), A.rowPtr, A.colVal, B, ldb, beta, C, ldc)
            C
        end
        function mm2!(transa::SparseChar,
                      transb::SparseChar,
                      alpha::Number,
                      A::CuSparseMatrixCSC{$elty},
                      B::CuMatrix{$elty},
                      beta::Number,
                      C::CuMatrix{$elty},
                      index::SparseChar)
            if transb == 'C'
                throw(ArgumentError("B^H is not supported"))
            elseif transb == 'T' && transa != 'N'
                throw(ArgumentError("When using B^T, A can be neither transposed nor adjointed"))
            end
            ctransa = 'N'
            if transa == 'N'
                ctransa = 'T'
            end
            desc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, index)
            k,m = A.dims
            n = size(C)[2]
            if ctransa == 'N' && transb == 'N'
                chkmmdims(B,C,k,n,m,n)
            elseif ctransa == 'N' && transb != 'N'
                chkmmdims(B,C,n,k,m,n)
            elseif ctransa != 'N' && transb == 'N'
                chkmmdims(B,C,m,n,k,n)
            elseif ctransa != 'N' && transb != 'N'
                chkmmdims(B,C,n,m,k,n)
            end
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            $fname(handle(),
                   ctransa, transb, m, n, k, nnz(A), alpha, desc,
                   nonzeros(A), A.colPtr, rowvals(A), B, ldb, beta, C, ldc)
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
                      X::CuMatrix{$elty},
                      index::SparseChar)
            desc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, uplo, diag, index)
            m,n = A.dims
            if m != n
                 throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mb = div(m,A.blockDim)
            mX,nX = size(X)
            if transxy == 'N' && (mX != m)
                throw(DimensionMismatch(""))
            end
            if transxy != 'N' && (nX != m)
                throw(DimensionMismatch(""))
            end
            ldx = max(1,stride(X,2))
            info = bsrsm2Info_t[0]
            cusparseCreateBsrsm2Info(info)
            @workspace size=@argout(
                    $bname(handle(), A.dir, transa, transxy,
                           mb, nX, nnz(A), desc, nonzeros(A), A.rowPtr,
                           A.colVal, A.blockDim, info[],
                           out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), A.dir, transa, transxy,
                           mb, nX, nnz(A), desc, nonzeros(A), A.rowPtr,
                           A.colVal, A.blockDim, info[],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXbsrsm2_zeroPivot(handle(), info[], posit)
                    if posit[] >= 0
                        error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                    end
                    $sname(handle(), A.dir, transa, transxy, mb,
                           nX, nnz(A), alpha, desc, nonzeros(A), A.rowPtr,
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
                      X::CuMatrix{$elty},
                      index::SparseChar)
            desc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, uplo, diag, index)
            m,n = A.dims
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mX,nX = size(X)
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
            @workspace size=@argout(
                    $bname(handle(), 0, transa, transxy,
                           m, nX, nnz(A), alpha, desc, nonzeros(A), A.rowPtr,
                           A.colVal, X, ldx, info[], CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                           out(Ref{UInt64}(1)))
                )[] buffer->begin
                    $aname(handle(), 0, transa, transxy,
                           m, nX, nnz(A), alpha, desc, nonzeros(A), A.rowPtr,
                           A.colVal, X, ldx, info[],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXcsrsm2_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                    end
                    $sname(handle(), 0, transa, transxy, m,
                           nX, nnz(A), alpha, desc, nonzeros(A), A.rowPtr,
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
                      X::CuMatrix{$elty},
                      index::SparseChar)
            ctransa = 'N'
            cuplo = 'U'
            if transa == 'N'
                ctransa = 'T'
            end
            if uplo == 'U'
                cuplo = 'L'
            end
            desc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, cuplo, diag, index)
            n,m = A.dims
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($n,$m)!"))
            end
            mX,nX = size(X)
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
            @workspace size=@argout(
                    $bname(handle(), 0, ctransa, transxy,
                           m, nX, nnz(A), alpha, desc, nonzeros(A), A.colPtr,
                           rowvals(A), X, ldx, info[], CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                           out(Ref{UInt64}(1)))
                )[] buffer->begin
                    $aname(handle(), 0, ctransa, transxy,
                           m, nX, nnz(A), alpha, desc, nonzeros(A), A.colPtr,
                           rowvals(A), X, ldx, info[],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXcsrsm2_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                    end
                    $sname(handle(), 0, ctransa, transxy, m,
                           nX, nnz(A), alpha, desc, nonzeros(A), A.colPtr,
                           rowvals(A), X, ldx, info[],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyCsrsm2Info(info[])
            X
        end
    end
end

for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        function sm2(transa::SparseChar,
                     transxy::SparseChar,
                     uplo::SparseChar,
                     diag::SparseChar,
                     alpha::Number,
                     A::CuSparseMatrix{$elty},
                     X::CuMatrix{$elty},
                     index::SparseChar)
            sm2!(transa,transxy,uplo,diag,alpha,A,copy(X),index)
        end
        function sm2(transa::SparseChar,
                     transxy::SparseChar,
                     uplo::SparseChar,
                     diag::SparseChar,
                     A::CuSparseMatrix{$elty},
                     X::CuMatrix{$elty},
                     index::SparseChar)
            sm2!(transa,transxy,uplo,diag,one($elty),A,copy(X),index)
        end
        function sm2(transa::SparseChar,
                     transxy::SparseChar,
                     uplo::SparseChar,
                     A::CuSparseMatrix{$elty},
                     X::CuMatrix{$elty},
                     index::SparseChar)
            sm2!(transa,transxy,uplo,'N',one($elty),A,copy(X),index)
        end
    end
end
