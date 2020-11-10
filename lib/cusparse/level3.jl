# sparse linear algebra functions that perform operations between sparse and (usually tall)
# dense matrices

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

# bsrsm2
for (bname,aname,sname,elty) in ((:cusparseSbsrsm2_bufferSize, :cusparseSbsrsm2_analysis, :cusparseSbsrsm2_solve, :Float32),
                                 (:cusparseDbsrsm2_bufferSize, :cusparseDbsrsm2_analysis, :cusparseDbsrsm2_solve, :Float64),
                                 (:cusparseCbsrsm2_bufferSize, :cusparseCbsrsm2_analysis, :cusparseCbsrsm2_solve, :ComplexF32),
                                 (:cusparseZbsrsm2_bufferSize, :cusparseZbsrsm2_analysis, :cusparseZbsrsm2_solve, :ComplexF64))
    @eval begin
        function bsrsm2!(transa::SparseChar,
                         transxy::SparseChar,
                         alpha::Number,
                         A::CuSparseMatrixBSR{$elty},
                         X::CuMatrix{$elty},
                         index::SparseChar)
            desc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT, index)
            m,n = A.dims
            if m != n
                throw(DimensionMismatch("A must be square!"))
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
        function bsrsm2(transa::SparseChar,
                        transxy::SparseChar,
                        alpha::Number,
                        A::CuSparseMatrixBSR{$elty},
                        X::CuMatrix{$elty},
                        index::SparseChar)
            bsrsm2!(transa,transxy,alpha,A,copy(X),index)
        end
    end
end

# csrsm2
for (bname,aname,sname,elty) in ((:cusparseScsrsm2_bufferSizeExt, :cusparseScsrsm2_analysis, :cusparseScsrsm2_solve, :Float32),
                                 (:cusparseDcsrsm2_bufferSizeExt, :cusparseDcsrsm2_analysis, :cusparseDcsrsm2_solve, :Float64),
                                 (:cusparseCcsrsm2_bufferSizeExt, :cusparseCcsrsm2_analysis, :cusparseCcsrsm2_solve, :ComplexF32),
                                 (:cusparseZcsrsm2_bufferSizeExt, :cusparseZcsrsm2_analysis, :cusparseZcsrsm2_solve, :ComplexF64))
    @eval begin
        function csrsm2!(transa::SparseChar,
                         transxy::SparseChar,
                         alpha::Number,
                         A::CuSparseMatrixCSR{$elty},
                         X::CuMatrix{$elty},
                         index::SparseChar)
            desc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT, index)
            m,n = A.dims
            if m != n
                throw(DimensionMismatch("A must be square!"))
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
        function csrsm2(transa::SparseChar,
                        transxy::SparseChar,
                        alpha::Number,
                        A::CuSparseMatrixCSR{$elty},
                        X::CuMatrix{$elty},
                        index::SparseChar)
            csrsm2!(transa,transxy,alpha,A,copy(X),index)
        end
    end
end
