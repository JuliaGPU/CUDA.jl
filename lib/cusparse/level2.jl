# sparse linear algebra functions that perform operations between sparse matrices and dense
# vectors

export sv2!, sv2, mv!

"""
    mv!(transa::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, X::CuVector, beta::BlasFloat, Y::CuVector, index::SparseChar)

Performs `Y = alpha * op(A) * X + beta * Y`, where `op` can be nothing (`transa = N`),
tranpose (`transa = T`) or conjugate transpose (`transa = C`). `X` is a sparse vector, and
`Y` is dense.
"""
mv!(transa::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, X::CuVector, beta::BlasFloat, Y::CuVector, index::SparseChar)

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
            desc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, index)
            m,n = A.dims
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
    sv2!(transa::SparseChar, uplo::SparseChar, diag::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, X::CuVector, index::SparseChar)

Performs `X = alpha * op(A) \\ X`, where `op` can be nothing (`transa = N`), tranpose
(`transa = T`) or conjugate transpose (`transa = C`). `X` is a dense vector, and `uplo`
tells `sv2!` which triangle of the block sparse matrix `A` to reference.
If the triangle has unit diagonal, set `diag` to 'U'.
"""
sv2!(transa::SparseChar, uplo::SparseChar, diag::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, X::CuVector, index::SparseChar)

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
            desc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, uplo, diag, index)
            m,n = A.dims
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mb = div(m,A.blockDim)
            mX = length(X)
            if mX != m
                throw(DimensionMismatch("X must have length $m, but has length $mX"))
            end
            info = bsrsv2Info_t[0]
            cusparseCreateBsrsv2Info(info)
            @workspace size=@argout(
                    $bname(handle(), A.dir, transa, mb, nnz(A),
                           desc, nonzeros(A), A.rowPtr, A.colVal, A.blockDim,
                           info[1], out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), A.dir, transa, mb, nnz(A),
                           desc, nonzeros(A), A.rowPtr, A.colVal, A.blockDim,
                           info[1], CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXbsrsv2_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                    end
                    $sname(handle(), A.dir, transa, mb, nnz(A),
                           alpha, desc, nonzeros(A), A.rowPtr, A.colVal,
                           A.blockDim, info[1], X, X,
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyBsrsv2Info(info[1])
            X
        end
    end
end

# csrsv2
for (bname,aname,sname,elty) in ((:cusparseScsrsv2_bufferSize, :cusparseScsrsv2_analysis, :cusparseScsrsv2_solve, :Float32),
                                 (:cusparseDcsrsv2_bufferSize, :cusparseDcsrsv2_analysis, :cusparseDcsrsv2_solve, :Float64),
                                 (:cusparseCcsrsv2_bufferSize, :cusparseCcsrsv2_analysis, :cusparseCcsrsv2_solve, :ComplexF32),
                                 (:cusparseZcsrsv2_bufferSize, :cusparseZcsrsv2_analysis, :cusparseZcsrsv2_solve, :ComplexF64))
    @eval begin
        function sv2!(transa::SparseChar,
                      uplo::SparseChar,
                      diag::SparseChar,
                      alpha::Number,
                      A::CuSparseMatrixCSR{$elty},
                      X::CuVector{$elty},
                      index::SparseChar)
            desc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, uplo, diag, index)
            m,n = A.dims
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mX = length(X)
            if mX != m
                throw(DimensionMismatch("First dimension of A, $m, and of X, $(size(X)[1]) must match"))
            end
            info = csrsv2Info_t[0]
            cusparseCreateCsrsv2Info(info)
            @workspace size=@argout(
                    $bname(handle(), transa, m, nnz(A),
                           desc, nonzeros(A), A.rowPtr, A.colVal, info[1],
                           out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), transa, m, nnz(A),
                           desc, nonzeros(A), A.rowPtr, A.colVal, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXcsrsv2_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                    end
                    $sname(handle(), transa, m,
                           nnz(A), alpha, desc, nonzeros(A), A.rowPtr,
                           A.colVal, info[1], X, X,
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyCsrsv2Info(info[1])
            X
        end
    end
end

# cscsv2
for (bname,aname,sname,elty) in ((:cusparseScsrsv2_bufferSize, :cusparseScsrsv2_analysis, :cusparseScsrsv2_solve, :Float32),
                                 (:cusparseDcsrsv2_bufferSize, :cusparseDcsrsv2_analysis, :cusparseDcsrsv2_solve, :Float64),
                                 (:cusparseCcsrsv2_bufferSize, :cusparseCcsrsv2_analysis, :cusparseCcsrsv2_solve, :ComplexF32),
                                 (:cusparseZcsrsv2_bufferSize, :cusparseZcsrsv2_analysis, :cusparseZcsrsv2_solve, :ComplexF64))
    @eval begin
        function sv2!(transa::SparseChar,
                      uplo::SparseChar,
                      diag::SparseChar,
                      alpha::Number,
                      A::CuSparseMatrixCSC{$elty},
                      X::CuVector{$elty},
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
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mX = length(X)
            if mX != m
                throw(DimensionMismatch("First dimension of A, $m, and of X, $(size(X)[1]) must match"))
            end
            info = csrsv2Info_t[0]
            cusparseCreateCsrsv2Info(info)
            @workspace size=@argout(
                    $bname(handle(), ctransa, m, nnz(A),
                           desc, nonzeros(A), A.colPtr, rowvals(A), info[1],
                           out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), ctransa, m, nnz(A),
                           desc, nonzeros(A), A.colPtr, rowvals(A), info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXcsrsv2_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                    end
                    $sname(handle(), ctransa, m,
                           nnz(A), alpha, desc, nonzeros(A), A.colPtr,
                           rowvals(A), info[1], X, X,
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyCsrsv2Info(info[1])
            X
        end
    end
end

for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        function sv2(transa::SparseChar,
                     uplo::SparseChar,
                     diag::SparseChar,
                     alpha::Number,
                     A::CuSparseMatrix{$elty},
                     X::CuVector{$elty},
                     index::SparseChar)
            sv2!(transa,uplo,diag,alpha,A,copy(X),index)
        end
        function sv2(transa::SparseChar,
                     uplo::SparseChar,
                     diag::SparseChar,
                     A::CuSparseMatrix{$elty},
                     X::CuVector{$elty},
                     index::SparseChar)
            sv2!(transa,uplo,diag,one($elty),A,copy(X),index)
        end
        function sv2(transa::SparseChar,
                     uplo::SparseChar,
                     A::CuSparseMatrix{$elty},
                     X::CuVector{$elty},
                     index::SparseChar)
            sv2!(transa,uplo,'N',one($elty),A,copy(X),index)
        end
    end
end
