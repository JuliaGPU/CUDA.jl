# wrappers of low-level functionality

import LinearAlgebra: SingularException, HermOrSym, AbstractTriangular, *, +, -, \, mul!


## essentials

function cusparseCreate()
    handle = Ref{cusparseHandle_t}()
    res = @retry_reclaim err->isequal(err, CUSPARSE_STATUS_ALLOC_FAILED) ||
                              isequal(err, CUSPARSE_STATUS_NOT_INITIALIZED) begin
        unsafe_cusparseCreate(handle)
    end
    if res != CUSPARSE_STATUS_SUCCESS
         throw_api_error(res)
    end
    handle[]
end

function cusparseGetProperty(property::libraryPropertyType)
    value_ref = Ref{Cint}()
    cusparseGetProperty(property, value_ref)
    value_ref[]
end

version() = VersionNumber(cusparseGetProperty(CUDA.MAJOR_VERSION),
                          cusparseGetProperty(CUDA.MINOR_VERSION),
                          cusparseGetProperty(CUDA.PATCH_LEVEL))


## level 1 functions

export axpyi!, axpyi, sctr!, sctr, gthr!, gthr, gthrz!, roti!, roti

"""
    axpyi!(alpha::BlasFloat, X::CuSparseVector, Y::CuVector, index::SparseChar)

Computes `alpha * X + Y` for sparse `X` and dense `Y`.
"""
axpyi!(alpha::BlasFloat, X::CuSparseVector, Y::CuVector, index::SparseChar)

for (fname,elty) in ((:cusparseSaxpyi, :Float32),
                     (:cusparseDaxpyi, :Float64),
                     (:cusparseCaxpyi, :ComplexF32),
                     (:cusparseZaxpyi, :ComplexF64))
    @eval begin
        function axpyi!(alpha::$elty,
                        X::CuSparseVector{$elty},
                        Y::CuVector{$elty},
                        index::SparseChar)
            $fname(handle(), X.nnz, [alpha], X.nzVal, X.iPtr, Y, index)
            Y
        end
        function axpyi(alpha::$elty,
                       X::CuSparseVector{$elty},
                       Y::CuVector{$elty},
                       index::SparseChar)
            axpyi!(alpha,X,copy(Y),index)
        end
        function axpyi(X::CuSparseVector{$elty},
                       Y::CuVector{$elty},
                       index::SparseChar)
            axpyi!(one($elty),X,copy(Y),index)
        end
    end
end

"""
    gthr!(X::CuSparseVector, Y::CuVector, index::SparseChar)

Sets the nonzero elements of `X` equal to the nonzero elements of `Y` at the same indices.
"""
gthr!(X::CuSparseVector, Y::CuVector, index::SparseChar)
for (fname,elty) in ((:cusparseSgthr, :Float32),
                     (:cusparseDgthr, :Float64),
                     (:cusparseCgthr, :ComplexF32),
                     (:cusparseZgthr, :ComplexF64))
    @eval begin
        function gthr!(X::CuSparseVector{$elty},
                       Y::CuVector{$elty},
                       index::SparseChar)
            $fname(handle(), X.nnz, Y, X.nzVal, X.iPtr, index)
            X
        end
        function gthr(X::CuSparseVector{$elty},
                      Y::CuVector{$elty},
                      index::SparseChar)
            gthr!(copy(X),Y,index)
        end
    end
end

"""
    gthrz!(X::CuSparseVector, Y::CuVector, index::SparseChar)

Sets the nonzero elements of `X` equal to the nonzero elements of `Y` at the same indices, and zeros out those elements of `Y`.
"""
gthrz!(X::CuSparseVector, Y::CuVector, index::SparseChar)
for (fname,elty) in ((:cusparseSgthrz, :Float32),
                     (:cusparseDgthrz, :Float64),
                     (:cusparseCgthrz, :ComplexF32),
                     (:cusparseZgthrz, :ComplexF64))
    @eval begin
        function gthrz!(X::CuSparseVector{$elty},
                        Y::CuVector{$elty},
                        index::SparseChar)
            $fname(handle(), X.nnz, Y, X.nzVal, X.iPtr, index)
            X,Y
        end
        function gthrz(X::CuSparseVector{$elty},
                       Y::CuVector{$elty},
                       index::SparseChar)
            gthrz!(copy(X),copy(Y),index)
        end
    end
end

"""
    roti!(X::CuSparseVector, Y::CuVector, c::BlasFloat, s::BlasFloat, index::SparseChar)

Performs the Givens rotation specified by `c` and `s` to sparse `X` and dense `Y`.
"""
roti!(X::CuSparseVector, Y::CuVector, c::BlasFloat, s::BlasFloat, index::SparseChar)
for (fname,elty) in ((:cusparseSroti, :Float32),
                     (:cusparseDroti, :Float64))
    @eval begin
        function roti!(X::CuSparseVector{$elty},
                       Y::CuVector{$elty},
                       c::$elty,
                       s::$elty,
                       index::SparseChar)
            $fname(handle(), X.nnz, X.nzVal, X.iPtr, Y, [c], [s], index)
            X,Y
        end
        function roti(X::CuSparseVector{$elty},
                      Y::CuVector{$elty},
                      c::$elty,
                      s::$elty,
                      index::SparseChar)
            roti!(copy(X),copy(Y),c,s,index)
        end
    end
end

"""
    sctr!(X::CuSparseVector, Y::CuVector, index::SparseChar)

Set `Y[:] = X[:]` for dense `Y` and sparse `X`.
"""
sctr!(X::CuSparseVector, Y::CuVector, index::SparseChar)
for (fname,elty) in ((:cusparseSsctr, :Float32),
                     (:cusparseDsctr, :Float64),
                     (:cusparseCsctr, :ComplexF32),
                     (:cusparseZsctr, :ComplexF64))
    @eval begin
        function sctr!(X::CuSparseVector{$elty},
                       Y::CuVector{$elty},
                       index::SparseChar)
            $fname(handle(), X.nnz, X.nzVal, X.iPtr, Y, index)
            Y
        end
        function sctr(X::CuSparseVector{$elty},
                      index::SparseChar)
            sctr!(X, CUDA.zeros($elty, X.dims[1]),index)
        end
    end
end


## level 2 functions

export sv2!, sv2, sv

"""
    mv!(transa::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, X::CuVector, beta::BlasFloat, Y::CuVector, index::SparseChar)

Performs `Y = alpha * op(A) *X + beta * Y`, where `op` can be nothing (`transa = N`), tranpose (`transa = T`)
or conjugate transpose (`transa = C`). `X` is a sparse vector, and `Y` is dense.
"""
mv!(transa::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, X::CuVector, beta::BlasFloat, Y::CuVector, index::SparseChar)
for (fname,elty) in ((:cusparseSbsrmv, :Float32),
                     (:cusparseDbsrmv, :Float64),
                     (:cusparseCbsrmv, :ComplexF32),
                     (:cusparseZbsrmv, :ComplexF64))
    @eval begin
        function mv!(transa::SparseChar,
                     alpha::$elty,
                     A::CuSparseMatrixBSR{$elty},
                     X::CuVector{$elty},
                     beta::$elty,
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
                   A.nnz, [alpha], desc, A.nzVal, A.rowPtr,
                   A.colVal, A.blockDim, X, [beta], Y)
            Y
        end
    end
end

"""
    sv2!(transa::SparseChar, uplo::SparseChar, alpha::BlasFloat, A::CuSparseMatrixBSR, X::CuVector, index::SparseChar)

Performs `X = alpha * op(A) \\ X `, where `op` can be nothing (`transa = N`), tranpose (`transa = T`)
or conjugate transpose (`transa = C`). `X` is a dense vector, and `uplo` tells `sv2!` which triangle
of the block sparse matrix `A` to reference.
"""
sv2!(transa::SparseChar, uplo::SparseChar, alpha::BlasFloat, A::CuSparseMatrixBSR, X::CuVector, index::SparseChar)
# bsrsv2
for (bname,aname,sname,elty) in ((:cusparseSbsrsv2_bufferSize, :cusparseSbsrsv2_analysis, :cusparseSbsrsv2_solve, :Float32),
                                 (:cusparseDbsrsv2_bufferSize, :cusparseDbsrsv2_analysis, :cusparseDbsrsv2_solve, :Float64),
                                 (:cusparseCbsrsv2_bufferSize, :cusparseCbsrsv2_analysis, :cusparseCbsrsv2_solve, :ComplexF32),
                                 (:cusparseZbsrsv2_bufferSize, :cusparseZbsrsv2_analysis, :cusparseZbsrsv2_solve, :ComplexF64))
    @eval begin
        function sv2!(transa::SparseChar,
                      uplo::SparseChar,
                      alpha::$elty,
                      A::CuSparseMatrixBSR{$elty},
                      X::CuVector{$elty},
                      index::SparseChar)
            desc   = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, uplo, CUSPARSE_DIAG_TYPE_NON_UNIT, index)
            m,n      = A.dims
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
                    $bname(handle(), A.dir, transa, mb, A.nnz,
                           desc, A.nzVal, A.rowPtr, A.colVal, A.blockDim,
                           info[1], out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), A.dir, transa, mb, A.nnz,
                           desc, A.nzVal, A.rowPtr, A.colVal, A.blockDim,
                           info[1], CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXbsrsv2_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                    end
                    $sname(handle(), A.dir, transa, mb, A.nnz,
                           [alpha], desc, A.nzVal, A.rowPtr, A.colVal,
                           A.blockDim, info[1], X, X,
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyBsrsv2Info(info[1])
            X
        end
    end
end

for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        function sv2(transa::SparseChar,
                     uplo::SparseChar,
                     alpha::$elty,
                     A::CuSparseMatrix{$elty},
                     X::CuVector{$elty},
                     index::SparseChar)
            sv2!(transa,uplo,alpha,A,copy(X),index)
        end
        function sv2(transa::SparseChar,
                     uplo::SparseChar,
                     A::CuSparseMatrix{$elty},
                     X::CuVector{$elty},
                     index::SparseChar)
            sv2!(transa,uplo,one($elty),A,copy(X),index)
        end
        function sv2(transa::SparseChar,
                     alpha::$elty,
                     A::AbstractTriangular,
                     X::CuVector{$elty},
                     index::SparseChar)
            uplo = 'U'
            if istril(A)
                uplo = 'L'
            end
            sv2!(transa,uplo,alpha,A.data,copy(X),index)
        end
        function sv2(transa::SparseChar,
                     A::AbstractTriangular,
                     X::CuVector{$elty},
                     index::SparseChar)
            uplo = 'U'
            if istril(A)
                uplo = 'L'
            end
            sv2!(transa,uplo,one($elty),A.data,copy(X),index)
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
                      alpha::$elty,
                      A::CuSparseMatrixCSR{$elty},
                      X::CuVector{$elty},
                      index::SparseChar)
            desc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, uplo, CUSPARSE_DIAG_TYPE_NON_UNIT, index)
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
                    $bname(handle(), transa, m, A.nnz,
                           desc, A.nzVal, A.rowPtr, A.colVal, info[1],
                           out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), transa, m, A.nnz,
                           desc, A.nzVal, A.rowPtr, A.colVal, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXcsrsv2_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                    end
                    $sname(handle(), transa, m,
                           A.nnz, [alpha], desc, A.nzVal, A.rowPtr,
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
                      alpha::$elty,
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
            desc   = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, cuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, index)
            n,m      = A.dims
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
                    $bname(handle(), ctransa, m, A.nnz,
                           desc, A.nzVal, A.colPtr, A.rowVal, info[1],
                           out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), ctransa, m, A.nnz,
                           desc, A.nzVal, A.colPtr, A.rowVal, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXcsrsv2_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                    end
                    $sname(handle(), ctransa, m,
                           A.nnz, [alpha], desc, A.nzVal, A.colPtr,
                           A.rowVal, info[1], X, X,
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyCsrsv2Info(info[1])
            X
        end
    end
end


## level 3 functions

export mm2!, mm2

"""
    mm2!(transa::SparseChar, transb::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, B::CuMatrix, beta::BlasFloat, C::CuMatrix, index::SparseChar)

Multiply the sparse matrix `A` by the dense matrix `B`, filling in dense matrix `C`.
`C = alpha*op(A)*op(B) + beta*C`. `op(A)` can be nothing (`transa = N`), transpose
(`transa = T`), or conjugate transpose (`transa = C`), and similarly for `op(B)` and
`transb`.
"""
mm2!(transa::SparseChar, transb::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, B::CuMatrix, beta::BlasFloat, C::CuMatrix, index::SparseChar)
for (fname,elty) in ((:cusparseSbsrmm, :Float32),
                     (:cusparseDbsrmm, :Float64),
                     (:cusparseCbsrmm, :ComplexF32),
                     (:cusparseZbsrmm, :ComplexF64))
    @eval begin
        function mm2!(transa::SparseChar,
                      transb::SparseChar,
                      alpha::$elty,
                      A::CuSparseMatrixBSR{$elty},
                      B::CuMatrix{$elty},
                      beta::$elty,
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
                   transa, transb, mb, n, kb, A.nnz,
                   [alpha], desc, A.nzVal,A.rowPtr, A.colVal,
                   A.blockDim, B, ldb, [beta], C, ldc)
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
                      alpha::$elty,
                      A::CuSparseMatrixCSR{$elty},
                      B::CuMatrix{$elty},
                      beta::$elty,
                      C::CuMatrix{$elty},
                      index::SparseChar)
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
                   transa, transb, m, n, k, A.nnz, [alpha], desc,
                   A.nzVal, A.rowPtr, A.colVal, B, ldb, [beta], C, ldc)
            C
        end
        function mm2!(transa::SparseChar,
                      transb::SparseChar,
                      alpha::$elty,
                      A::CuSparseMatrixCSC{$elty},
                      B::CuMatrix{$elty},
                      beta::$elty,
                      C::CuMatrix{$elty},
                      index::SparseChar)
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
                   ctransa, transb, m, n, k, A.nnz, [alpha], desc,
                   A.nzVal, A.colPtr, A.rowVal, B, ldb, [beta], C, ldc)
            C
        end
    end
end

for elty in (:Float32,:Float64,:ComplexF32,:ComplexF64)
    @eval begin
        function mm2(transa::SparseChar,
                     transb::SparseChar,
                     alpha::$elty,
                     A::Union{CuSparseMatrixCSR{$elty},CuSparseMatrixCSC{$elty},CuSparseMatrixBSR{$elty}},
                     B::CuMatrix{$elty},
                     beta::$elty,
                     C::CuMatrix{$elty},
                     index::SparseChar)
            mm2!(transa,transb,alpha,A,B,beta,copy(C),index)
        end
        function mm2(transa::SparseChar,
                     transb::SparseChar,
                     A::Union{CuSparseMatrixCSR{$elty},CuSparseMatrixCSC{$elty},CuSparseMatrixBSR{$elty}},
                     B::CuMatrix{$elty},
                     beta::$elty,
                     C::CuMatrix{$elty},
                     index::SparseChar)
            mm2(transa,transb,one($elty),A,B,beta,C,index)
        end
        function mm2(transa::SparseChar,
                     transb::SparseChar,
                     A::Union{CuSparseMatrixCSR{$elty},CuSparseMatrixCSC{$elty},CuSparseMatrixBSR{$elty}},
                     B::CuMatrix{$elty},
                     C::CuMatrix{$elty},
                     index::SparseChar)
            mm2(transa,transb,one($elty),A,B,one($elty),C,index)
        end
        function mm2(transa::SparseChar,
                     transb::SparseChar,
                     alpha::$elty,
                     A::Union{CuSparseMatrixCSR{$elty},CuSparseMatrixCSC{$elty},CuSparseMatrixBSR{$elty}},
                     B::CuMatrix{$elty},
                     index::SparseChar)
            m = transa == 'N' ? size(A)[1] : size(A)[2]
            n = transb == 'N' ? size(B)[2] : size(B)[1]
            mm2(transa,transb,alpha,A,B,zero($elty),CUDA.zeros($elty,(m,n)),index)
        end
        function mm2(transa::SparseChar,
                     transb::SparseChar,
                     A::Union{CuSparseMatrixCSR{$elty},CuSparseMatrixCSC{$elty},CuSparseMatrixBSR{$elty}},
                     B::CuMatrix{$elty},
                     index::SparseChar)
            m = transa == 'N' ? size(A)[1] : size(A)[2]
            n = transb == 'N' ? size(B)[2] : size(B)[1]
            mm2(transa,transb,one($elty),A,B,zero($elty),CUDA.zeros($elty,(m,n)),index)
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
                         alpha::$elty,
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
                           mb, nX, A.nnz, desc, A.nzVal, A.rowPtr,
                           A.colVal, A.blockDim, info[],
                           out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), A.dir, transa, transxy,
                           mb, nX, A.nnz, desc, A.nzVal, A.rowPtr,
                           A.colVal, A.blockDim, info[],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXbsrsm2_zeroPivot(handle(), info[], posit)
                    if posit[] >= 0
                        error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                    end
                    $sname(handle(), A.dir, transa, transxy, mb,
                           nX, A.nnz, [alpha], desc, A.nzVal, A.rowPtr,
                           A.colVal, A.blockDim, info[], X, ldx, X, ldx,
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyBsrsm2Info(info[1])
            X
        end
        function bsrsm2(transa::SparseChar,
                        transxy::SparseChar,
                        alpha::$elty,
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
                         alpha::$elty,
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
                           m, nX, A.nnz, [alpha], desc, A.nzVal, A.rowPtr,
                           A.colVal, X, ldx, info[], CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                           out(Ref{UInt64}(1)))
                )[] buffer->begin
                    $aname(handle(), 0, transa, transxy,
                           m, nX, A.nnz, [alpha], desc, A.nzVal, A.rowPtr,
                           A.colVal, X, ldx, info[],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXcsrsm2_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                    end
                    $sname(handle(), 0, transa, transxy, m,
                           nX, A.nnz, [alpha], desc, A.nzVal, A.rowPtr,
                           A.colVal, X, ldx, info[],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyCsrsm2Info(info[])
            X
        end
        function csrsm2(transa::SparseChar,
                        transxy::SparseChar,
                        alpha::$elty,
                        A::CuSparseMatrixCSR{$elty},
                        X::CuMatrix{$elty},
                        index::SparseChar)
            csrsm2!(transa,transxy,alpha,A,copy(X),index)
        end
    end
end


## preconditioners

export ic02!, ic02, ilu02!, ilu02

"""
    ic02!(A::CuSparseMatrix, index::SparseChar)

Incomplete Cholesky factorization with no pivoting.
Preserves the sparse layout of matrix `A`.
"""
ic02!(A::CuSparseMatrix, index::SparseChar)
for (bname,aname,sname,elty) in ((:cusparseScsric02_bufferSize, :cusparseScsric02_analysis, :cusparseScsric02, :Float32),
                                 (:cusparseDcsric02_bufferSize, :cusparseDcsric02_analysis, :cusparseDcsric02, :Float64),
                                 (:cusparseCcsric02_bufferSize, :cusparseCcsric02_analysis, :cusparseCcsric02, :ComplexF32),
                                 (:cusparseZcsric02_bufferSize, :cusparseZcsric02_analysis, :cusparseZcsric02, :ComplexF64))
    @eval begin
        function ic02!(A::CuSparseMatrixCSR{$elty},
                       index::SparseChar)
            desc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, index)
            m,n = A.dims
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = csric02Info_t[0]
            cusparseCreateCsric02Info(info)
            @workspace size=@argout(
                    $bname(handle(), m, A.nnz, desc,
                           A.nzVal, A.rowPtr, A.colVal, info[1],
                           out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), m, A.nnz, desc,
                           A.nzVal, A.rowPtr, A.colVal, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXcsric02_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                    end
                    $sname(handle(), m, A.nnz,
                           desc, A.nzVal, A.rowPtr, A.colVal, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyCsric02Info(info[1])
            A
        end
    end
end

# cscic02
for (bname,aname,sname,elty) in ((:cusparseScsric02_bufferSize, :cusparseScsric02_analysis, :cusparseScsric02, :Float32),
                                 (:cusparseDcsric02_bufferSize, :cusparseDcsric02_analysis, :cusparseDcsric02, :Float64),
                                 (:cusparseCcsric02_bufferSize, :cusparseCcsric02_analysis, :cusparseCcsric02, :ComplexF32),
                                 (:cusparseZcsric02_bufferSize, :cusparseZcsric02_analysis, :cusparseZcsric02, :ComplexF64))
    @eval begin
        function ic02!(A::CuSparseMatrixCSC{$elty},
                       index::SparseChar)
            desc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, index)
            m,n = A.dims
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = csric02Info_t[0]
            cusparseCreateCsric02Info(info)
            @workspace size=@argout(
                    $bname(handle(), m, A.nnz, desc,
                           A.nzVal, A.colPtr, A.rowVal, info[1],
                           out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), m, A.nnz, desc,
                           A.nzVal, A.colPtr, A.rowVal, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXcsric02_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                    end
                    $sname(handle(), m, A.nnz,
                           desc, A.nzVal, A.colPtr, A.rowVal, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyCsric02Info(info[1])
            A
        end
    end
end

"""
    ilu02!(A::CuSparseMatrix, index::SparseChar)

Incomplete LU factorization with no pivoting.
Preserves the sparse layout of matrix `A`.
"""
ilu02!(A::CuSparseMatrix, index::SparseChar)
for (bname,aname,sname,elty) in ((:cusparseScsrilu02_bufferSize, :cusparseScsrilu02_analysis, :cusparseScsrilu02, :Float32),
                                 (:cusparseDcsrilu02_bufferSize, :cusparseDcsrilu02_analysis, :cusparseDcsrilu02, :Float64),
                                 (:cusparseCcsrilu02_bufferSize, :cusparseCcsrilu02_analysis, :cusparseCcsrilu02, :ComplexF32),
                                 (:cusparseZcsrilu02_bufferSize, :cusparseZcsrilu02_analysis, :cusparseZcsrilu02, :ComplexF64))
    @eval begin
        function ilu02!(A::CuSparseMatrixCSR{$elty},
                        index::SparseChar)
            desc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, index)
            m,n = A.dims
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = csrilu02Info_t[0]
            cusparseCreateCsrilu02Info(info)
            @workspace size=@argout(
                    $bname(handle(), m, A.nnz, desc,
                           A.nzVal, A.rowPtr, A.colVal, info[1],
                           out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), m, A.nnz, desc,
                           A.nzVal, A.rowPtr, A.colVal, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXcsrilu02_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        error("Structural zero in A at ($(posit[]),$(posit[])))")
                    end
                    $sname(handle(), m, A.nnz,
                           desc, A.nzVal, A.rowPtr, A.colVal, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyCsrilu02Info(info[1])
            A
        end
    end
end

# cscilu02
for (bname,aname,sname,elty) in ((:cusparseScsrilu02_bufferSize, :cusparseScsrilu02_analysis, :cusparseScsrilu02, :Float32),
                                 (:cusparseDcsrilu02_bufferSize, :cusparseDcsrilu02_analysis, :cusparseDcsrilu02, :Float64),
                                 (:cusparseCcsrilu02_bufferSize, :cusparseCcsrilu02_analysis, :cusparseCcsrilu02, :ComplexF32),
                                 (:cusparseZcsrilu02_bufferSize, :cusparseZcsrilu02_analysis, :cusparseZcsrilu02, :ComplexF64))
    @eval begin
        function ilu02!(A::CuSparseMatrixCSC{$elty},
                        index::SparseChar)
            desc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, index)
            m,n = A.dims
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = csrilu02Info_t[0]
            cusparseCreateCsrilu02Info(info)
            @workspace size=@argout(
                    $bname(handle(), m, A.nnz, desc,
                           A.nzVal, A.colPtr, A.rowVal, info[1],
                           out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), m, A.nnz, desc,
                           A.nzVal, A.colPtr, A.rowVal, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXcsrilu02_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        error("Structural zero in A at ($(posit[]),$(posit[])))")
                    end
                    $sname(handle(), m, A.nnz,
                           desc, A.nzVal, A.colPtr, A.rowVal, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyCsrilu02Info(info[1])
            A
        end
    end
end

# bsric02
for (bname,aname,sname,elty) in ((:cusparseSbsric02_bufferSize, :cusparseSbsric02_analysis, :cusparseSbsric02, :Float32),
                                 (:cusparseDbsric02_bufferSize, :cusparseDbsric02_analysis, :cusparseDbsric02, :Float64),
                                 (:cusparseCbsric02_bufferSize, :cusparseCbsric02_analysis, :cusparseCbsric02, :ComplexF32),
                                 (:cusparseZbsric02_bufferSize, :cusparseZbsric02_analysis, :cusparseZbsric02, :ComplexF64))
    @eval begin
        function ic02!(A::CuSparseMatrixBSR{$elty},
                       index::SparseChar)
            desc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT, index)
            m,n = A.dims
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mb = div(m,A.blockDim)
            info = bsric02Info_t[0]
            cusparseCreateBsric02Info(info)
            @workspace size=@argout(
                    $bname(handle(), A.dir, mb, A.nnz, desc,
                           A.nzVal, A.rowPtr, A.colVal, A.blockDim, info[1],
                           out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), A.dir, mb, A.nnz, desc,
                           A.nzVal, A.rowPtr, A.colVal, A.blockDim, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXbsric02_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                    end
                    $sname(handle(), A.dir, mb, A.nnz, desc,
                           A.nzVal, A.rowPtr, A.colVal, A.blockDim, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyBsric02Info(info[1])
            A
        end
    end
end

# bsrilu02
for (bname,aname,sname,elty) in ((:cusparseSbsrilu02_bufferSize, :cusparseSbsrilu02_analysis, :cusparseSbsrilu02, :Float32),
                                 (:cusparseDbsrilu02_bufferSize, :cusparseDbsrilu02_analysis, :cusparseDbsrilu02, :Float64),
                                 (:cusparseCbsrilu02_bufferSize, :cusparseCbsrilu02_analysis, :cusparseCbsrilu02, :ComplexF32),
                                 (:cusparseZbsrilu02_bufferSize, :cusparseZbsrilu02_analysis, :cusparseZbsrilu02, :ComplexF64))
    @eval begin
        function ilu02!(A::CuSparseMatrixBSR{$elty},
                        index::SparseChar)
            desc = CuMatrixDescriptor(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT, index)
            m,n = A.dims
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mb = div(m,A.blockDim)
            info = bsrilu02Info_t[0]
            cusparseCreateBsrilu02Info(info)
            @workspace size=@argout(
                    $bname(handle(), A.dir, mb, A.nnz, desc,
                           A.nzVal, A.rowPtr, A.colVal, A.blockDim, info[1],
                           out(Ref{Cint}(1)))
                )[] buffer->begin
                    $aname(handle(), A.dir, mb, A.nnz, desc,
                           A.nzVal, A.rowPtr, A.colVal, A.blockDim, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                    posit = Ref{Cint}(1)
                    cusparseXbsrilu02_zeroPivot(handle(), info[1], posit)
                    if posit[] >= 0
                        error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                    end
                    $sname(handle(), A.dir, mb, A.nnz, desc,
                           A.nzVal, A.rowPtr, A.colVal, A.blockDim, info[1],
                           CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                end
            cusparseDestroyBsrilu02Info(info[1])
            A
        end
    end
end

for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        function ilu02(A::CuSparseMatrix{$elty},
                       index::SparseChar)
            ilu02!(copy(A),index)
        end
        function ic02(A::CuSparseMatrix{$elty},
                      index::SparseChar)
            ic02!(copy(A),index)
        end
        function ilu02(A::HermOrSym{$elty,CuSparseMatrix{$elty}},
                       index::SparseChar)
            ilu02!(copy(A.data),index)
        end
        function ic02(A::HermOrSym{$elty,CuSparseMatrix{$elty}},
                      index::SparseChar)
            ic02!(copy(A.data),index)
        end
    end
end


# SpMV

function mv!(
    transa::SparseChar, 
    alpha::T, 
    A::CuSparseMatrixCSR{T}, 
    X::CuVector{T}, 
    beta::T, 
    Y::CuVector{T}, 
    index::SparseChar
) where {T}

    cusparseSpMV(
        handle(),
        transa,
        [one(T)],
        CuSparseMatrixDescriptor(A),
        CuDenseVectorDescriptor(X),
        [zero(T)],
        CuDenseVectorDescriptor(Y),
        T,
        CUSPARSE_MV_ALG_DEFAULT,
        CU_NULL
    )

    Y

end
