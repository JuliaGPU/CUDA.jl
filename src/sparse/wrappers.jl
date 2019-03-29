# wrappers of the low-level CUSPARSE functionality
#
# TODO: move raw ccall wrappers to libcusparse.jl

#utilities
import LinearAlgebra: SingularException, HermOrSym, AbstractTriangular, *, +, -, \, mul!

export switch2csr, switch2csc, switch2bsr, switch2hyb
export axpyi!, axpyi, sctr!, sctr, gthr!, gthr, gthrz!, grthrz, roti!, roti
export doti!, doti, dotci!, dotci
export mv!, mv, sv2!, sv2, sv_analysis, sv_solve!, sv
export mm2!, mm2, mm!, mm, sm_analysis, sm_solve, sm
export geam, gemm
export ic0!, ic0, ic02!, ic02, ilu0!, ilu0, ilu02!, ilu02
export gtsv!, gtsv, gtsv_nopivot!, gtsv_nopivot, gtsvStridedBatch!, gtsvStridedBatch


# Level 1 CUSPARSE functions

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
            cuind = cusparseindex(index)
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Ptr{$elty}, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{$elty}, cusparseIndexBase_t),
                              handle(), X.nnz, [alpha], X.nzVal, X.iPtr,
                              Y, cuind)
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
    doti!(X::CuSparseVector, Y::CuVector, index::SparseChar)

Computes `dot(X,Y)` for sparse `X` and dense `Y`, without conjugation. 
"""
function doti!(X::CuSparseVector, Y::CuVector, index::SparseChar) end

"""
    dotci!(X::CuSparseVector, Y::CuVector, index::SparseChar)

Computes `dot(X,conj(Y))` for sparse `X` and dense `Y`. 
"""
function dotci!(X::CuSparseVector, Y::CuVector, index::SparseChar) end
for (jname,fname,elty) in ((:doti, :cusparseSdoti, :Float32),
                           (:doti, :cusparseDdoti, :Float64),
                           (:doti, :cusparseCdoti, :ComplexF32),
                           (:doti, :cusparseZdoti, :ComplexF64),
                           (:dotci, :cusparseCdotci, :ComplexF32),
                           (:dotci, :cusparseZdotci, :ComplexF64))
    @eval begin
        function $jname(X::CuSparseVector{$elty},
                        Y::CuVector{$elty},
                        index::SparseChar)
            dot = Ref{$elty}(1)
            cuind = cusparseindex(index)
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{$elty}, Ptr{$elty}, cusparseIndexBase_t),
                              handle(), X.nnz, X.nzVal, X.iPtr,
                              Y, dot, cuind)
            return dot[]
        end
    end
end

"""
    gthr!(X::CuSparseVector, Y::CuVector, index::SparseChar)

Sets the nonzero elements of `X` equal to the nonzero elements of `Y` at the same indices.
"""
function gthr!(X::CuSparseVector, Y::CuVector, index::SparseChar) end
for (fname,elty) in ((:cusparseSgthr, :Float32),
                     (:cusparseDgthr, :Float64),
                     (:cusparseCgthr, :ComplexF32),
                     (:cusparseZgthr, :ComplexF64))
    @eval begin
        function gthr!(X::CuSparseVector{$elty},
                       Y::CuVector{$elty},
                       index::SparseChar)
            cuind = cusparseindex(index)
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, CuPtr{$elty}, CuPtr{$elty},
                               CuPtr{Cint}, cusparseIndexBase_t), handle(),
                              X.nnz, Y, X.nzVal, X.iPtr, cuind)
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
function gthrz!(X::CuSparseVector, Y::CuVector, index::SparseChar) end
for (fname,elty) in ((:cusparseSgthrz, :Float32),
                     (:cusparseDgthrz, :Float64),
                     (:cusparseCgthrz, :ComplexF32),
                     (:cusparseZgthrz, :ComplexF64))
    @eval begin
        function gthrz!(X::CuSparseVector{$elty},
                        Y::CuVector{$elty},
                        index::SparseChar)
            cuind = cusparseindex(index)
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, CuPtr{$elty}, CuPtr{$elty},
                               CuPtr{Cint}, cusparseIndexBase_t), handle(),
                              X.nnz, Y, X.nzVal, X.iPtr, cuind)
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
function roti!(X::CuSparseVector, Y::CuVector, c::BlasFloat, s::BlasFloat, index::SparseChar) end
for (fname,elty) in ((:cusparseSroti, :Float32),
                     (:cusparseDroti, :Float64))
    @eval begin
        function roti!(X::CuSparseVector{$elty},
                       Y::CuVector{$elty},
                       c::$elty,
                       s::$elty,
                       index::SparseChar)
            cuind = cusparseindex(index)
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, CuPtr{$elty}, CuPtr{$Cint},
                               CuPtr{$elty}, Ptr{$elty}, Ptr{$elty}, cusparseIndexBase_t),
                              handle(), X.nnz, X.nzVal, X.iPtr, Y, [c], [s], cuind)
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
function sctr!(X::CuSparseVector, Y::CuVector, index::SparseChar) end

for (fname,elty) in ((:cusparseSsctr, :Float32),
                     (:cusparseDsctr, :Float64),
                     (:cusparseCsctr, :ComplexF32),
                     (:cusparseZsctr, :ComplexF64))
    @eval begin
        function sctr!(X::CuSparseVector{$elty},
                       Y::CuVector{$elty},
                       index::SparseChar)
            cuind = cusparseindex(index)
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{$elty}, cusparseIndexBase_t),
                              handle(), X.nnz, X.nzVal, X.iPtr,
                              Y, cuind)
            Y
        end
        function sctr(X::CuSparseVector{$elty},
                      index::SparseChar)
            sctr!(X, cuzeros($elty, X.dims[1]),index)
        end
    end
end

## level 2 functions

"""
    mv!(transa::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, X::CuVector, beta::BlasFloat, Y::CuVector, index::SparseChar)

Performs `Y = alpha * op(A) *X + beta * Y`, where `op` can be nothing (`transa = N`), tranpose (`transa = T`)
or conjugate transpose (`transa = C`). `X` is a sparse vector, and `Y` is dense.
"""
function mv!(transa::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, X::CuVector,
             beta::BlasFloat, Y::CuVector, index::SparseChar) end
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
            cudir = cusparsedir(A.dir)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            mb = div(m,A.blockDim)
            nb = div(n,A.blockDim)
            if transa == 'N'
                chkmvdims(X,n,Y,m)
            end
            if transa == 'T' || transa == 'C'
                chkmvdims(X,m,Y,n)
            end
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t,
                               cusparseOperation_t, Cint, Cint, Cint,
                               Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               CuPtr{$elty}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                               CuPtr{$elty}, Ptr{$elty}, CuPtr{$elty}),
                              handle(), cudir, cutransa, mb, nb,
                              A.nnz, [alpha], Ref(cudesc), A.nzVal, A.rowPtr,
                              A.colVal, A.blockDim, X, [beta], Y)
            Y
        end
    end
end

for (fname,elty) in ((:cusparseScsrmv, :Float32),
                     (:cusparseDcsrmv, :Float64),
                     (:cusparseCcsrmv, :ComplexF32),
                     (:cusparseZcsrmv, :ComplexF64))
    @eval begin
        function mv!(transa::SparseChar,
                     alpha::$elty,
                     A::Union{CuSparseMatrixCSR{$elty},HermOrSym{$elty,CuSparseMatrixCSR{$elty}}},
                     X::CuVector{$elty},
                     beta::$elty,
                     Y::CuVector{$elty},
                     index::SparseChar)
            Mat     = A
            if typeof(A) <: HermOrSym
                 Mat = A.data
            end
            cutransa = cusparseop(transa)
            m,n = Mat.dims
            if transa == 'N'
                chkmvdims(X, n, Y, m)
            end
            if transa == 'T' || transa == 'C'
                chkmvdims(X, m, Y, n)
            end
            cudesc = getDescr(A,index)
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Cint, Cint, Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               CuPtr{$elty}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{$elty},
                               Ptr{$elty}, CuPtr{$elty}), handle(),
                               cutransa, m, n, Mat.nnz, [alpha], Ref(cudesc), Mat.nzVal,
                               Mat.rowPtr, Mat.colVal, X, [beta], Y)
            Y
        end
        function mv!(transa::SparseChar,
                     alpha::$elty,
                     A::Union{CuSparseMatrixCSC{$elty},HermOrSym{$elty,CuSparseMatrixCSC{$elty}}},
                     X::CuVector{$elty},
                     beta::$elty,
                     Y::CuVector{$elty},
                     index::SparseChar)
            Mat     = A
            if typeof(A) <: HermOrSym
                 Mat = A.data
            end
            ctransa  = 'N'
            if transa == 'N'
                ctransa = 'T'
            end
            cutransa = cusparseop(ctransa)
            cuind    = cusparseindex(index)
            cudesc   = getDescr(A,index)
            n,m      = Mat.dims
            if ctransa == 'N'
                chkmvdims(X,n,Y,m)
            end
            if ctransa == 'T' || ctransa == 'C'
                chkmvdims(X,m,Y,n)
            end
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Cint, Cint, Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               CuPtr{$elty}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{$elty},
                               Ptr{$elty}, CuPtr{$elty}), handle(),
                               cutransa, m, n, Mat.nnz, [alpha], Ref(cudesc),
                               Mat.nzVal, Mat.colPtr, Mat.rowVal, X, [beta], Y)
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
function sv2!(transa::SparseChar, uplo::SparseChar, alpha::BlasFloat, A::CuSparseMatrixBSR, X::CuVector, index::SparseChar) end
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
            cutransa = cusparseop(transa)
            cudir    = cusparsedir(A.dir)
            cuind    = cusparseindex(index)
            cuplo    = cusparsefill(uplo)
            cudesc   = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, cuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n      = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mb = div(m,A.blockDim)
            mX = length(X)
            if( mX != m )
                throw(DimensionMismatch("X must have length $m, but has length $mX"))
            end
            info = bsrsv2Info_t[0]
            cusparseCreateBsrsv2Info(info)
            bufSize = Ref{Cint}(1)
            @check ccall(($(string(bname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t,
                               cusparseOperation_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, Cint, bsrsv2Info_t, Ptr{Cint}),
                              handle(), cudir, cutransa, mb, A.nnz,
                              Ref(cudesc), A.nzVal, A.rowPtr, A.colVal,
                              A.blockDim, info[1], bufSize)
            buffer = cuzeros(UInt8, bufSize[])
            @check ccall(($(string(aname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t,
                               cusparseOperation_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, Cint, bsrsv2Info_t,
                               cusparseSolvePolicy_t, CuPtr{Cvoid}),
                              handle(), cudir, cutransa, mb, A.nnz,
                              Ref(cudesc), A.nzVal, A.rowPtr, A.colVal, A.blockDim,
                              info[1], CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            posit = Ref{Cint}(1)
            @check ccall((:cusparseXbsrsv2_zeroPivot, libcusparse),
                        cusparseStatus_t, (cusparseHandle_t, bsrsv2Info_t,
                        Ptr{Cint}), handle(), info[1], posit)
            if( posit[] >= 0 )
                throw(string("Structural/numerical zero in A at (",posit[],posit[],")"))
            end
            @check ccall(($(string(sname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t,
                               cusparseOperation_t, Cint, Cint, Ptr{$elty},
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, Cint, bsrsv2Info_t, CuPtr{$elty},
                               CuPtr{$elty}, cusparseSolvePolicy_t, CuPtr{Cvoid}),
                              handle(), cudir, cutransa, mb, A.nnz,
                              [alpha], Ref(cudesc), A.nzVal, A.rowPtr, A.colVal,
                              A.blockDim, info[1], X, X,
                              CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
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

"""
    sv_analysis(transa::SparseChar, typea::SparseChar, uplo::SparseChar, A::CuSparseMatrixCSR, index::SparseChar)

Perform preliminary analysis of sparse matrix `A` before doing a solve of the form `Y = op(A) \\ X`. `transa = N` for no
op, `transa = T` for transpose, and `transa = C` for conjugate transpose. `uplo` tells CUSPARSE which triangle of `A` to
reference, and `typea` whether `A` is a general matrix (`G`), symmetric (`S`), Hermitian (`H`), or triangular (`T`).
"""
function sv_analysis(transa::SparseChar, typea::SparseChar, uplo::SparseChar, A::CuSparseMatrixCSR, index::SparseChar) end

for (fname,elty) in ((:cusparseScsrsv_analysis, :Float32),
                     (:cusparseDcsrsv_analysis, :Float64),
                     (:cusparseCcsrsv_analysis, :ComplexF32),
                     (:cusparseZcsrsv_analysis, :ComplexF64))
    @eval begin
        function sv_analysis(transa::SparseChar,
                             typea::SparseChar,
                             uplo::SparseChar,
                             A::CuSparseMatrixCSR{$elty},
                             index::SparseChar)
            cutransa = cusparseop(transa)
            cuind    = cusparseindex(index)
            cutype   = cusparsetype(typea)
            cuuplo   = cusparsefill(uplo)
            cudesc   = cusparseMatDescr_t(cutype, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = cusparseSolveAnalysisInfo_t[0]
            cusparseCreateSolveAnalysisInfo(info)
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t},
                               CuPtr{$elty}, CuPtr{Cint}, CuPtr{Cint},
                               cusparseSolveAnalysisInfo_t), handle(),
                               cutransa, m, A.nnz, Ref(cudesc), A.nzVal,
                               A.rowPtr, A.colVal, info[1])
            info[1]
        end
    end
end

for (fname,elty) in ((:cusparseScsrsv_analysis, :Float32),
                     (:cusparseDcsrsv_analysis, :Float64),
                     (:cusparseCcsrsv_analysis, :ComplexF32),
                     (:cusparseZcsrsv_analysis, :ComplexF64))
    @eval begin
        function sv_analysis(transa::SparseChar,
                             typea::SparseChar,
                             uplo::SparseChar,
                             A::CuSparseMatrixCSC{$elty},
                             index::SparseChar)
            ctransa = 'N'
            if transa == 'N'
                ctransa = 'T'
            end
            cuplo = 'U'
            if uplo == 'U'
                cuplo = 'L'
            end
            cutransa = cusparseop(ctransa)
            cuind    = cusparseindex(index)
            cutype   = cusparsetype(typea)
            cuuplo   = cusparsefill(cuplo)
            cudesc   = cusparseMatDescr_t(cutype, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            n,m      = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = cusparseSolveAnalysisInfo_t[0]
            cusparseCreateSolveAnalysisInfo(info)
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t},
                               CuPtr{$elty}, CuPtr{Cint}, CuPtr{Cint},
                               cusparseSolveAnalysisInfo_t), handle(),
                               cutransa, m, A.nnz, Ref(cudesc), A.nzVal,
                               A.colPtr, A.rowVal, info[1])
            info[1]
        end
    end
end

"""
    sv_solve!(transa::SparseChar, uplo::SparseChar, alpha::BlasFloat, A::CuSparseMatrixCSR, X::CuVector, Y::CuVector, info::cusparseSolveAnalysisInfo_t, index::SparseChar)

Solve the problem `Y = op(A)\\ alpha*X`. The operation is determined by `transa`. `info` is
the output of [`sv_analysis`](@ref). The arguments `transa`, `uplo`, and `index` must be the same
between the `analysis` and `solve` steps. 
"""
function sv_solve!(transa::SparseChar, uplo::SparseChar, alpha::BlasFloat, A::CuSparseMatrixCSR, X::CuVector, Y::CuVector, info::cusparseSolveAnalysisInfo_t, index::SparseChar) end
for (fname,elty) in ((:cusparseScsrsv_solve, :Float32),
                     (:cusparseDcsrsv_solve, :Float64),
                     (:cusparseCcsrsv_solve, :ComplexF32),
                     (:cusparseZcsrsv_solve, :ComplexF64))
    @eval begin
        function sv_solve!(transa::SparseChar,
                           uplo::SparseChar,
                           alpha::$elty,
                           A::CuSparseMatrixCSR{$elty},
                           X::CuVector{$elty},
                           Y::CuVector{$elty},
                           info::cusparseSolveAnalysisInfo_t,
                           index::SparseChar)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cuuplo = cusparsefill(uplo)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_TRIANGULAR, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( size(X)[1] != m )
                throw(DimensionMismatch("First dimension of A, $m, and of X, $(size(X)[1]) must match"))
            end
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Ptr{$elty}, Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, cusparseSolveAnalysisInfo_t,
                               CuPtr{$elty}, CuPtr{$elty}), handle(),
                               cutransa, m, [alpha], Ref(cudesc), A.nzVal,
                               A.rowPtr, A.colVal, info, X, Y)
            Y
        end
    end
end

for (fname,elty) in ((:cusparseScsrsv_solve, :Float32),
                     (:cusparseDcsrsv_solve, :Float64),
                     (:cusparseCcsrsv_solve, :ComplexF32),
                     (:cusparseZcsrsv_solve, :ComplexF64))

    @eval begin
        function sv_solve!(transa::SparseChar,
                           uplo::SparseChar,
                           alpha::$elty,
                           A::CuSparseMatrixCSC{$elty},
                           X::CuVector{$elty},
                           Y::CuVector{$elty},
                           info::cusparseSolveAnalysisInfo_t,
                           index::SparseChar)
            ctransa = 'N'
            if transa == 'N'
                ctransa = 'T'
            end
            cuplo = 'U'
            if uplo == 'U'
                cuplo = 'L'
            end
            cutransa = cusparseop(ctransa)
            cuind    = cusparseindex(index)
            cuuplo   = cusparsefill(cuplo)
            cudesc   = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_TRIANGULAR, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            n,m      = A.dims
            if( size(X)[1] != m )
                throw(DimensionMismatch("First dimension of A, $m, and of X, $(size(X)[1]) must match"))
            end
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Ptr{$elty}, Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, cusparseSolveAnalysisInfo_t,
                               CuPtr{$elty}, CuPtr{$elty}), handle(),
                               cutransa, m, [alpha], Ref(cudesc), A.nzVal,
                               A.colPtr, A.rowVal, info, X, Y)
            Y
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
            cutransa  = cusparseop(transa)
            cuind     = cusparseindex(index)
            cuuplo    = cusparsefill(uplo)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mX = length(X)
            if( mX != m )
                throw(DimensionMismatch("First dimension of A, $m, and of X, $(size(X)[1]) must match"))
            end
            info = csrsv2Info_t[0]
            cusparseCreateCsrsv2Info(info)
            bufSize = Ref{Cint}(1)
            @check ccall(($(string(bname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, csrsv2Info_t, Ptr{Cint}),
                              handle(), cutransa, m, A.nnz,
                              Ref(cudesc), A.nzVal, A.rowPtr, A.colVal,
                              info[1], bufSize)
            buffer = cuzeros(UInt8, bufSize[])
            @check ccall(($(string(aname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, csrsv2Info_t, cusparseSolvePolicy_t,
                               CuPtr{Cvoid}), handle(), cutransa, m, A.nnz,
                               Ref(cudesc), A.nzVal, A.rowPtr, A.colVal, info[1],
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            posit = Ref{Cint}(1)
            @check ccall((:cusparseXcsrsv2_zeroPivot, libcusparse),
                        cusparseStatus_t, (cusparseHandle_t, csrsv2Info_t,
                        Ptr{Cint}), handle(), info[1], posit)
            if( posit[] >= 0 )
                throw(string("Structural/numerical zero in A at (",posit[],posit[],")"))
            end
            @check ccall(($(string(sname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Cint, Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               CuPtr{$elty}, CuPtr{Cint}, CuPtr{Cint}, csrsv2Info_t,
                               CuPtr{$elty}, CuPtr{$elty}, cusparseSolvePolicy_t,
                               CuPtr{Cvoid}), handle(), cutransa, m,
                               A.nnz, [alpha], Ref(cudesc), A.nzVal, A.rowPtr,
                               A.colVal, info[1], X, X,
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
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
            cutransa = cusparseop(ctransa)
            cuind    = cusparseindex(index)
            cuuplo   = cusparsefill(cuplo)
            cudesc   = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            n,m      = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mX = length(X)
            if( mX != m )
                throw(DimensionMismatch("First dimension of A, $m, and of X, $(size(X)[1]) must match"))
            end
            info = csrsv2Info_t[0]
            cusparseCreateCsrsv2Info(info)
            bufSize = Ref{Cint}(1)
            @check ccall(($(string(bname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, csrsv2Info_t, Ptr{Cint}),
                              handle(), cutransa, m, A.nnz,
                              Ref(cudesc), A.nzVal, A.colPtr, A.rowVal,
                              info[1], bufSize)
            buffer = cuzeros(UInt8, bufSize[])
            @check ccall(($(string(aname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, csrsv2Info_t, cusparseSolvePolicy_t,
                               CuPtr{Cvoid}), handle(), cutransa, m, A.nnz,
                               Ref(cudesc), A.nzVal, A.colPtr, A.rowVal, info[1],
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            posit = Ref{Cint}(1)
            @check ccall((:cusparseXcsrsv2_zeroPivot, libcusparse),
                        cusparseStatus_t, (cusparseHandle_t, csrsv2Info_t,
                        Ptr{Cint}), handle(), info[1], posit)
            if( posit[] >= 0 )
                throw(string("Structural/numerical zero in A at (",posit[],posit[],")"))
            end
            @check ccall(($(string(sname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Cint, Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               CuPtr{$elty}, CuPtr{Cint}, CuPtr{Cint}, csrsv2Info_t,
                               CuPtr{$elty}, CuPtr{$elty}, cusparseSolvePolicy_t,
                               CuPtr{Cvoid}), handle(), cutransa, m,
                               A.nnz, [alpha], Ref(cudesc), A.nzVal, A.colPtr,
                               A.rowVal, info[1], X, X,
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            cusparseDestroyCsrsv2Info(info[1])
            X
        end
    end
end

for (fname,elty) in ((:cusparseShybmv, :Float32),
                     (:cusparseDhybmv, :Float64),
                     (:cusparseChybmv, :ComplexF32),
                     (:cusparseZhybmv, :ComplexF64))
    @eval begin
        function mv!(transa::SparseChar,
                     alpha::$elty,
                     A::CuSparseMatrixHYB{$elty},
                     X::CuVector{$elty},
                     beta::$elty,
                     Y::CuVector{$elty},
                     index::SparseChar)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if transa == 'N'
                chkmvdims(X,n,Y,m)
            end
            if transa == 'T' || transa == 'C'
                chkmvdims(X,m,Y,n)
            end
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t,
                               Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               cusparseHybMat_t, CuPtr{$elty},
                               Ptr{$elty}, CuPtr{$elty}), handle(),
                               cutransa, [alpha], Ref(cudesc), A.Mat, X, [beta], Y)
            Y
        end
    end
end

for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        function mv(transa::SparseChar,
                    alpha::$elty,
                    A::Union{CuSparseMatrix{$elty},CompressedSparse{$elty}},
                    X::CuVector{$elty},
                    beta::$elty,
                    Y::CuVector{$elty},
                    index::SparseChar)
            mv!(transa,alpha,A,X,beta,copy(Y),index)
        end
        function mv(transa::SparseChar,
                    alpha::$elty,
                    A::Union{CuSparseMatrix{$elty},CompressedSparse{$elty}},
                    X::CuVector{$elty},
                    Y::CuVector{$elty},
                    index::SparseChar)
            mv(transa,alpha,A,X,one($elty),Y,index)
        end
        function mv(transa::SparseChar,
                    A::Union{CuSparseMatrix{$elty},CompressedSparse{$elty}},
                    X::CuVector{$elty},
                    beta::$elty,
                    Y::CuVector{$elty},
                    index::SparseChar)
            mv(transa,one($elty),A,X,beta,Y,index)
        end
        function mv(transa::SparseChar,
                    A::Union{CuSparseMatrix{$elty},CompressedSparse{$elty}},
                    X::CuVector{$elty},
                    Y::CuVector{$elty},
                    index::SparseChar)
            mv(transa,one($elty),A,X,one($elty),Y,index)
        end
        function mv(transa::SparseChar,
                    alpha::$elty,
                    A::Union{CuSparseMatrix{$elty},CompressedSparse{$elty}},
                    X::CuVector{$elty},
                    index::SparseChar)
            mv(transa,alpha,A,X,zero($elty),cuzeros($elty,size(A)[1]),index)
        end
        function mv(transa::SparseChar,
                    A::Union{CuSparseMatrix{$elty},CompressedSparse{$elty}},
                    X::CuVector{$elty},
                    index::SparseChar)
            mv(transa,one($elty),A,X,zero($elty),cuzeros($elty,size(A)[1]),index)
        end
    end
end

for (fname,elty) in ((:cusparseShybsv_analysis, :Float32),
                     (:cusparseDhybsv_analysis, :Float64),
                     (:cusparseChybsv_analysis, :ComplexF32),
                     (:cusparseZhybsv_analysis, :ComplexF64))
    @eval begin
        function sv_analysis(transa::SparseChar,
                             typea::SparseChar,
                             uplo::SparseChar,
                             A::CuSparseMatrixHYB{$elty},
                             index::SparseChar)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cuuplo = cusparsefill(uplo)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_TRIANGULAR, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = cusparseSolveAnalysisInfo_t[0]
            cusparseCreateSolveAnalysisInfo(info)
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t,
                               Ptr{cusparseMatDescr_t}, cusparseHybMat_t,
                               cusparseSolveAnalysisInfo_t),
                              handle(), cutransa, Ref(cudesc), A.Mat,
                              info[1])
            info[1]
        end
    end
end

for (fname,elty) in ((:cusparseShybsv_solve, :Float32),
                     (:cusparseDhybsv_solve, :Float64),
                     (:cusparseChybsv_solve, :ComplexF32),
                     (:cusparseZhybsv_solve, :ComplexF64))
    @eval begin
        function sv_solve!(transa::SparseChar,
                           uplo::SparseChar,
                           alpha::$elty,
                           A::CuSparseMatrixHYB{$elty},
                           X::CuVector{$elty},
                           Y::CuVector{$elty},
                           info::cusparseSolveAnalysisInfo_t,
                           index::SparseChar)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cuuplo = cusparsefill(uplo)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_TRIANGULAR, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( size(X)[1] != m )
                throw(DimensionMismatch("First dimension of A, $m, and of X, $(size(X)[1]) must match"))
            end
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t,
                               Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               cusparseHybMat_t, cusparseSolveAnalysisInfo_t,
                               CuPtr{$elty}, CuPtr{$elty}), handle(),
                               cutransa, [alpha], Ref(cudesc), A.Mat, info, X, Y)
            Y
        end
    end
end

"""
    sv(transa::SparseChar, typea::SparseChar, uplo::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, X::CuVector, index::SparseChar)

Solve the problem `op(A)\\ alpha*X`.
"""
function sv(transa::SparseChar, typea::SparseChar, uplo::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, X::CuVector, index::SparseChar) end

for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        function sv_solve(transa::SparseChar,
                          uplo::SparseChar,
                          alpha::$elty,
                          A::CuSparseMatrix{$elty},
                          X::CuVector{$elty},
                          info::cusparseSolveAnalysisInfo_t,
                          index::SparseChar)
            Y = similar(X)
            sv_solve!(transa, uplo, alpha, A, X, Y, info, index)
        end
        function sv(transa::SparseChar,
                    typea::SparseChar,
                    uplo::SparseChar,
                    alpha::$elty,
                    A::CuSparseMatrix{$elty},
                    X::CuVector{$elty},
                    index::SparseChar)
            info = sv_analysis(transa,typea,uplo,A,index)
            sv_solve(transa,uplo,alpha,A,X,info,index)
        end
        function sv(transa::SparseChar,
                    typea::SparseChar,
                    uplo::SparseChar,
                    A::CuSparseMatrix{$elty},
                    X::CuVector{$elty},
                    index::SparseChar)
            info = sv_analysis(transa,typea,uplo,A,index)
            sv_solve(transa,uplo,one($elty),A,X,info,index)
        end
        function sv(transa::SparseChar,
                    A::AbstractTriangular,
                    X::CuVector{$elty},
                    index::SparseChar)
            uplo = 'U'
            if istril(A)
                uplo = 'L'
            end
            info = sv_analysis(transa,'T',uplo,A.data,index)
            sv_solve(transa,uplo,one($elty),A.data,X,info,index)
        end
        function sv_analysis(transa::SparseChar,
                             typea::SparseChar,
                             uplo::SparseChar,
                             A::HermOrSym{$elty},
                             index::SparseChar)
            sv_analysis(transa,typea,uplo,A.data,index)
        end
    end
end

## level 3 functions

"""
    mm2!(transa::SparseChar, transb::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, B::CuMatrix, beta::BlasFloat, C::CuMatrix, index::SparseChar)

Multiply the sparse matrix `A` by the dense matrix `B`, filling in dense matrix `C`.
`C = alpha*op(A)*op(B) + beta*C`. `op(A)` can be nothing (`transa = N`), transpose 
(`transa = T`), or conjugate transpose (`transa = C`), and similarly for `op(B)` and
`transb`.
"""
function mm2!(transa::SparseChar, transb::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, B::CuMatrix, beta::BlasFloat, C::CuMatrix, index::SparseChar) end
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
            cutransa = cusparseop(transa)
            cutransb = cusparseop(transb)
            cudir = cusparsedir(A.dir)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
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
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t,
                               cusparseOperation_t, cusparseOperation_t, Cint,
                               Cint, Cint, Cint, Ptr{$elty},
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, Cint, CuPtr{$elty}, Cint, Ptr{$elty},
                               CuPtr{$elty}, Cint), handle(), cudir,
                               cutransa, cutransb, mb, n, kb, A.nnz,
                               [alpha], Ref(cudesc), A.nzVal,A.rowPtr, A.colVal,
                               A.blockDim, B, ldb, [beta], C, ldc)
            C
        end
    end
end

"""
    mm!(transa::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, B::CuMatrix, beta::BlasFloat, C::CuMatrix, index::SparseChar)

Multiply the sparse matrix `A` by the dense matrix `B`, filling in dense matrix `C`.
`C = alpha*op(A)*B + beta*C`. `op(A)` can be nothing (`transa = N`), transpose 
(`transa = T`), or conjugate transpose (`transa = C`).
"""
function mm!(transa::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, B::CuMatrix, beta::BlasFloat, C::CuMatrix, index::SparseChar) end
for (fname,elty) in ((:cusparseScsrmm, :Float32),
                     (:cusparseDcsrmm, :Float64),
                     (:cusparseCcsrmm, :ComplexF32),
                     (:cusparseZcsrmm, :ComplexF64))
    @eval begin
        function mm!(transa::SparseChar,
                     alpha::$elty,
                     A::Union{HermOrSym{$elty,CuSparseMatrixCSR{$elty}},CuSparseMatrixCSR{$elty}},
                     B::CuMatrix{$elty},
                     beta::$elty,
                     C::CuMatrix{$elty},
                     index::SparseChar)
            Mat     = A
            if typeof(A) <: HermOrSym
                 Mat = A.data
            end
            cutransa = cusparseop(transa)
            cuind    = cusparseindex(index)
            cudesc   = getDescr(A,index)
            m,k      = Mat.dims
            n        = size(C)[2]
            if transa == 'N'
                chkmmdims(B,C,k,n,m,n)
            else
                chkmmdims(B,C,m,n,k,n)
            end
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint, Cint,
                               Cint, Cint, Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               CuPtr{$elty}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{$elty},
                               Cint, Ptr{$elty}, CuPtr{$elty}, Cint),
                               handle(), cutransa, m, n, k, Mat.nnz,
                               [alpha], Ref(cudesc), Mat.nzVal, Mat.rowPtr,
                               Mat.colVal, B, ldb, [beta], C, ldc)
            C
        end
        function mm!(transa::SparseChar,
                     alpha::$elty,
                     A::Union{HermOrSym{$elty,CuSparseMatrixCSC{$elty}},CuSparseMatrixCSC{$elty}},
                     B::CuMatrix{$elty},
                     beta::$elty,
                     C::CuMatrix{$elty},
                     index::SparseChar)
            Mat     = A
            if typeof(A) <: HermOrSym
                Mat = A.data
            end
            ctransa = 'N'
            if transa == 'N'
                ctransa = 'T'
            end
            cutransa = cusparseop(ctransa)
            cuind    = cusparseindex(index)
            cudesc   = getDescr(A,index)
            k,m      = Mat.dims
            n        = size(C)[2]
            if ctransa == 'N'
                chkmmdims(B,C,k,n,m,n)
            else
                chkmmdims(B,C,m,n,k,n)
            end
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint, Cint,
                               Cint, Cint, Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               CuPtr{$elty}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{$elty},
                               Cint, Ptr{$elty}, CuPtr{$elty}, Cint),
                               handle(), cutransa, m, n, k, Mat.nnz,
                               [alpha], Ref(cudesc), Mat.nzVal, Mat.colPtr,
                               Mat.rowVal, B, ldb, [beta], C, ldc)
            C
        end
    end
end

for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        function mm(transa::SparseChar,
                    alpha::$elty,
                    A::CuSparseMatrix{$elty},
                    B::CuMatrix{$elty},
                    beta::$elty,
                    C::CuMatrix{$elty},
                    index::SparseChar)
            mm!(transa,alpha,A,B,beta,copy(C),index)
        end
        function mm(transa::SparseChar,
                    A::CuSparseMatrix{$elty},
                    B::CuMatrix{$elty},
                    beta::$elty,
                    C::CuMatrix{$elty},
                    index::SparseChar)
            mm(transa,one($elty),A,B,beta,C,index)
        end
        function mm(transa::SparseChar,
                    A::CuSparseMatrix{$elty},
                    B::CuMatrix{$elty},
                    C::CuMatrix{$elty},
                    index::SparseChar)
            mm(transa,one($elty),A,B,one($elty),C,index)
        end
        function mm(transa::SparseChar,
                    alpha::$elty,
                    A::CuSparseMatrix{$elty},
                    B::CuMatrix{$elty},
                    index::SparseChar)
            m = transa == 'N' ? size(A)[1] : size(A)[2]
            mm!(transa,alpha,A,B,zero($elty),cuzeros($elty,(m,size(B)[2])),index)
        end
        function mm(transa::SparseChar,
                    A::CuSparseMatrix{$elty},
                    B::CuMatrix{$elty},
                    index::SparseChar)
            m = transa == 'N' ? size(A)[1] : size(A)[2]
            mm!(transa,one($elty),A,B,zero($elty),cuzeros($elty,(m,size(B)[2])),index)
        end
        function mm(transa::SparseChar,
                    A::HermOrSym,
                    B::CuMatrix{$elty},
                    index::SparseChar)
            m = transa == 'N' ? size(A)[1] : size(A)[2]
            mm!(transa,one($elty),A.data,B,zero($elty),cuzeros($elty,(m,size(B)[2])),index)
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
            cutransa = cusparseop(transa)
            cutransb = cusparseop(transb)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
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
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t,
                               cusparseOperation_t, Cint, Cint, Cint, Cint,
                               Ptr{$elty}, Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, CuPtr{$elty}, Cint,
                               Ptr{$elty}, CuPtr{$elty}, Cint), handle(),
                               cutransa, cutransb, m, n, k, A.nnz, [alpha], Ref(cudesc),
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
            cutransa = cusparseop(ctransa)
            cutransb = cusparseop(transb)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
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
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t,
                               cusparseOperation_t, Cint, Cint, Cint, Cint,
                               Ptr{$elty}, Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, CuPtr{$elty}, Cint,
                               Ptr{$elty}, CuPtr{$elty}, Cint), handle(),
                               cutransa, cutransb, m, n, k, A.nnz, [alpha], Ref(cudesc),
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
            mm2(transa,transb,alpha,A,B,zero($elty),cuzeros($elty,(m,n)),index)
        end
        function mm2(transa::SparseChar,
                     transb::SparseChar,
                     A::Union{CuSparseMatrixCSR{$elty},CuSparseMatrixCSC{$elty},CuSparseMatrixBSR{$elty}},
                     B::CuMatrix{$elty},
                     index::SparseChar)
            m = transa == 'N' ? size(A)[1] : size(A)[2]
            n = transb == 'N' ? size(B)[2] : size(B)[1]
            mm2(transa,transb,one($elty),A,B,zero($elty),cuzeros($elty,(m,n)),index)
        end
    end
end

"""
    sm_analysis(transa::SparseChar, uplo::SparseChar, A::CuSparseMatrix, index::SparseChar)

Performs initial analysis step on sparse matrix `A` that will be used
in the solution of `Y = op(A)\\X`. `op(A)` is set by `transa` and can be one of
nothing (`transa = N`), transpose (`transa = T`), or conjugate transpose (`transa = C`).
"""
function sm_analysis(transa::SparseChar, uplo::SparseChar, A::CuSparseMatrix, index::SparseChar) end

for (fname,elty) in ((:cusparseScsrsm_analysis, :Float32),
                     (:cusparseDcsrsm_analysis, :Float64),
                     (:cusparseCcsrsm_analysis, :ComplexF32),
                     (:cusparseZcsrsm_analysis, :ComplexF64))
    @eval begin
        function sm_analysis(transa::SparseChar,
                             uplo::SparseChar,
                             A::CuSparseMatrixCSR{$elty},
                             index::SparseChar)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cuuplo = cusparsefill(uplo)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_TRIANGULAR, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( n != m )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = cusparseSolveAnalysisInfo_t[0]
            cusparseCreateSolveAnalysisInfo(info)
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, cusparseSolveAnalysisInfo_t),
                              handle(), cutransa, m, A.nnz, Ref(cudesc),
                              A.nzVal, A.rowPtr, A.colVal, info[1])
            info[1]
        end
        function sm_analysis(transa::SparseChar,
                             uplo::SparseChar,
                             A::CuSparseMatrixCSC{$elty},
                             index::SparseChar)
            ctransa = 'N'
            if transa == 'N'
                ctransa = 'T'
            end
            cuplo = 'U'
            if uplo == 'U'
                cuplo = 'L'
            end
            cutransa = cusparseop(ctransa)
            cuind    = cusparseindex(index)
            cuuplo   = cusparsefill(cuplo)
            cudesc   = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_TRIANGULAR, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            n,m      = A.dims
            if( n != m )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = cusparseSolveAnalysisInfo_t[0]
            cusparseCreateSolveAnalysisInfo(info)
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, cusparseSolveAnalysisInfo_t),
                              handle(), cutransa, m, A.nnz, Ref(cudesc),
                              A.nzVal, A.colPtr, A.rowVal, info[1])
            info[1]
        end
    end
end

"""
    sm_solve(transa::SparseChar, uplo::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, X::CuMatrix, info::cusparseSolveAnalysisInfo_t, index::SparseChar)

Solves `Y = op(A)\\alpha*X`.  `op(A)` is set by `transa` and can be one of
nothing (`transa = N`), transpose (`transa = T`), or conjugate transpose (`transa = C`).
`info` is the result of calling [`sm_analysis`](@ref) on `A`. `transa`, `uplo`, and `index` must
be the same as they were in [`sm_analysis`](@ref).
"""
function sm_solve(transa::SparseChar, uplo::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, X::CuMatrix, info::cusparseSolveAnalysisInfo_t, index::SparseChar) end

for (fname,elty) in ((:cusparseScsrsm_solve, :Float32),
                     (:cusparseDcsrsm_solve, :Float64),
                     (:cusparseCcsrsm_solve, :ComplexF32),
                     (:cusparseZcsrsm_solve, :ComplexF64))
    @eval begin
        function sm_solve(transa::SparseChar,
                          uplo::SparseChar,
                          alpha::$elty,
                          A::CuSparseMatrixCSR{$elty},
                          X::CuMatrix{$elty},
                          info::cusparseSolveAnalysisInfo_t,
                          index::SparseChar)
            cutransa = cusparseop(transa)
            cuind = cusparseindex(index)
            cuuplo = cusparsefill(uplo)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_TRIANGULAR, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,nA = A.dims
            mX,n = X.dims
            if( mX != m )
                throw(DimensionMismatch("First dimension of A, $m, and X, $mX must match"))
            end
            Y = similar(X)
            ldx = max(1,stride(X,2))
            ldy = max(1,stride(Y,2))
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint, Cint,
                               Ptr{$elty}, Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, cusparseSolveAnalysisInfo_t,
                               CuPtr{$elty}, Cint, CuPtr{$elty}, Cint),
                              handle(), cutransa, m, n, [alpha],
                              Ref(cudesc), A.nzVal, A.rowPtr, A.colVal, info, X, ldx,
                              Y, ldy)
            Y
        end
        function sm_solve(transa::SparseChar,
                          uplo::SparseChar,
                          alpha::$elty,
                          A::CuSparseMatrixCSC{$elty},
                          X::CuMatrix{$elty},
                          info::cusparseSolveAnalysisInfo_t,
                          index::SparseChar)
            ctransa = 'N'
            if transa == 'N'
                ctransa = 'T'
            end
            cuplo = 'U'
            if uplo == 'U'
                cuplo = 'L'
            end
            cutransa = cusparseop(ctransa)
            cuind    = cusparseindex(index)
            cuuplo   = cusparsefill(cuplo)
            cudesc   = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_TRIANGULAR, cuuplo, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,nA     = A.dims
            mX,n     = X.dims
            if( mX != m )
                throw(DimensionMismatch("First dimension of A, $m, and X, $mX must match"))
            end
            Y = similar(X)
            ldx = max(1,stride(X,2))
            ldy = max(1,stride(Y,2))
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint, Cint,
                               Ptr{$elty}, Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, cusparseSolveAnalysisInfo_t,
                               CuPtr{$elty}, Cint, CuPtr{$elty}, Cint),
                              handle(), cutransa, m, n, [alpha],
                              Ref(cudesc), A.nzVal, A.colPtr, A.rowVal, info, X, ldx,
                              Y, ldy)
            Y
        end
    end
end

"""
    sm(transa::SparseChar, uplo::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, B::CuMatrix, index::SparseChar)

Solve `C = op(A)\\alpha*B` where `A` is a sparse matrix and `B` is a dense matrix. `op(A)`
is set by `transa` and can be one of nothing (`transa = N`), transpose (`transa = T`),
or conjugate transpose (`transa = C`). `uplo` sets which triangle of `A` to reference.
"""
function sm(transa::SparseChar, uplo::SparseChar, alpha::BlasFloat, A::CuSparseMatrix, B::CuMatrix, index::SparseChar) end

for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        function sm(transa::SparseChar,
                    uplo::SparseChar,
                    alpha::$elty,
                    A::CuSparseMatrix{$elty},
                    B::CuMatrix{$elty},
                    index::SparseChar)
            info = sm_analysis(transa,uplo,A,index)
            sm_solve(transa,uplo,alpha,A,B,info,index)
        end
        function sm(transa::SparseChar,
                    uplo::SparseChar,
                    A::CuSparseMatrix{$elty},
                    B::CuMatrix{$elty},
                    index::SparseChar)
            info = sm_analysis(transa,uplo,A,index)
            sm_solve(transa,uplo,one($elty),A,B,info,index)
        end
        function sm(transa::SparseChar,
                    alpha::$elty,
                    A::AbstractTriangular,
                    B::CuMatrix{$elty},
                    index::SparseChar)
            uplo = 'U'
            if istril(A)
                uplo = 'L'
            end
            info = sm_analysis(transa,uplo,A.data,index)
            sm_solve(transa,uplo,alpha,A.data,B,info,index)
        end
        function sm(transa::SparseChar,
                    A::AbstractTriangular,
                    B::CuMatrix{$elty},
                    index::SparseChar)
            uplo = 'U'
            if istril(A)
                uplo = 'L'
            end
            info = sm_analysis(transa,uplo,A.data,index)
            sm_solve(transa,uplo,one($elty),A.data,B,info,index)
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
            cutransa  = cusparseop(transa)
            cutransxy = cusparseop(transxy)
            cudir = cusparsedir(A.dir)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square!"))
            end
            mb = div(m,A.blockDim)
            mX,nX = size(X)
            if( transxy == 'N' && (mX != m) )
                throw(DimensionMismatch(""))
            end
            if( transxy != 'N' && (nX != m) )
                throw(DimensionMismatch(""))
            end
            ldx = max(1,stride(X,2))
            info = bsrsm2Info_t[0]
            cusparseCreateBsrsm2Info(info)
            bufSize = Ref{Cint}(1)
            @check ccall(($(string(bname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t,
                               cusparseOperation_t, cusparseOperation_t, Cint,
                               Cint, Cint, Ptr{cusparseMatDescr_t},
                               CuPtr{$elty}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                               bsrsm2Info_t, Ptr{Cint}), handle(),
                               cudir, cutransa, cutransxy, mb, nX, A.nnz,
                               Ref(cudesc), A.nzVal, A.rowPtr, A.colVal,
                               A.blockDim, info[1], bufSize)
            buffer = cuzeros(UInt8, bufSize[])
            @check ccall(($(string(aname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t,
                               cusparseOperation_t, cusparseOperation_t, Cint,
                               Cint, Cint, Ptr{cusparseMatDescr_t},
                               CuPtr{$elty}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                               bsrsm2Info_t, cusparseSolvePolicy_t, CuPtr{Cvoid}),
                              handle(), cudir, cutransa, cutransxy,
                              mb, nX, A.nnz, Ref(cudesc), A.nzVal, A.rowPtr,
                              A.colVal, A.blockDim, info[1],
                              CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            posit = Ref{Cint}(1)
            @check ccall((:cusparseXbsrsm2_zeroPivot, libcusparse),
                        cusparseStatus_t, (cusparseHandle_t, bsrsm2Info_t,
                        Ptr{Cint}), handle(), info[1], posit)
            if( posit[] >= 0 )
                throw(string("Structural/numerical zero in A at (",posit[],posit[],")"))
            end
            @check ccall(($(string(sname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t,
                               cusparseOperation_t, cusparseOperation_t, Cint,
                               Cint, Cint, Ptr{$elty}, Ptr{cusparseMatDescr_t},
                               CuPtr{$elty}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                               bsrsm2Info_t, CuPtr{$elty}, Cint, CuPtr{$elty}, Cint,
                               cusparseSolvePolicy_t, CuPtr{Cvoid}),
                              handle(), cudir, cutransa, cutransxy, mb,
                              nX, A.nnz, [alpha], Ref(cudesc), A.nzVal, A.rowPtr,
                              A.colVal, A.blockDim, info[1], X, ldx, X, ldx,
                              CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
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

# extensions

"""
    geam(alpha::BlasFloat, A::CuSparseMatrix, beta::BlasFloat, B::CuSparseMatrix, indexA::SparseChar, indexB::SparseChar, indexC::SparseChar)

Solves `C = alpha * A + beta * B`. `A`, `B`, and `C` are all sparse.
"""
function geam(alpha::BlasFloat, A::CuSparseMatrix, beta::BlasFloat, B::CuSparseMatrix, indexA::SparseChar, indexB::SparseChar, indexC::SparseChar) end

for (fname,elty) in ((:cusparseScsrgeam, :Float32),
                     (:cusparseDcsrgeam, :Float64),
                     (:cusparseCcsrgeam, :ComplexF32),
                     (:cusparseZcsrgeam, :ComplexF64))
    @eval begin
        function geam(alpha::$elty,
                      A::CuSparseMatrixCSR{$elty},
                      beta::$elty,
                      B::CuSparseMatrixCSR{$elty},
                      indexA::SparseChar,
                      indexB::SparseChar,
                      indexC::SparseChar)
            cuinda = cusparseindex(indexA)
            cuindb = cusparseindex(indexB)
            cuindc = cusparseindex(indexB)
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            cudescb = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindb)
            cudescc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindc)
            mA,nA = A.dims
            mB,nB = B.dims
            if( (mA != mB) || (nA != nB) )
                throw(DimensionMismatch(""))
            end
            nnzC = Ref{Cint}(1)
            rowPtrC = cuzeros(Cint,mA+1)
            @check ccall((:cusparseXcsrgeamNnz,libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Cint, CuPtr{Cint},
                               CuPtr{Cint}, Ptr{cusparseMatDescr_t}, Cint, CuPtr{Cint},
                               CuPtr{Cint}, Ptr{cusparseMatDescr_t}, CuPtr{Cint},
                               Ptr{Cint}), handle(), mA, nA, Ref(cudesca),
                               A.nnz, A.rowPtr, A.colVal, Ref(cudescb), B.nnz,
                               B.rowPtr, B.colVal, Ref(cudescc), rowPtrC, nnzC)
            nnz = nnzC[]
            C = CuSparseMatrixCSR(rowPtrC, cuzeros(Cint,nnz), cuzeros($elty,nnz), nnz, A.dims)
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint, Ptr{$elty},
                               Ptr{cusparseMatDescr_t}, Cint, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, Ptr{$elty},
                               Ptr{cusparseMatDescr_t}, Cint, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, Ptr{cusparseMatDescr_t},
                               CuPtr{$elty}, CuPtr{Cint}, CuPtr{Cint}),
                              handle(), mA, nA, [alpha], Ref(cudesca),
                              A.nnz, A.nzVal, A.rowPtr, A.colVal, [beta],
                              Ref(cudescb), B.nnz, B.nzVal, B.rowPtr, B.colVal,
                              Ref(cudescc), C.nzVal, C.rowPtr, C.colVal)
            C
        end
        function geam(alpha::$elty,
                      A::CuSparseMatrixCSC{$elty},
                      beta::$elty,
                      B::CuSparseMatrixCSC{$elty},
                      indexA::SparseChar,
                      indexB::SparseChar,
                      indexC::SparseChar)
            cuinda = cusparseindex(indexA)
            cuindb = cusparseindex(indexB)
            cuindc = cusparseindex(indexB)
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            cudescb = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindb)
            cudescc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindc)
            mA,nA = A.dims
            mB,nB = B.dims
            if( (mA != mB) || (nA != nB) )
                throw(DimensionMismatch("A and B must have same dimensions!"))
            end
            nnzC = Ref{Cint}(1)
            rowPtrC = cuzeros(Cint, mA+1)
            @check ccall((:cusparseXcsrgeamNnz,libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Cint, CuPtr{Cint},
                               CuPtr{Cint}, Ptr{cusparseMatDescr_t}, Cint, CuPtr{Cint},
                               CuPtr{Cint}, Ptr{cusparseMatDescr_t}, CuPtr{Cint},
                               Ptr{Cint}), handle(), mA, nA, Ref(cudesca),
                               A.nnz, A.colPtr, A.rowVal, Ref(cudescb), B.nnz,
                               B.colPtr, B.rowVal, Ref(cudescc), rowPtrC, nnzC)
            nnz = nnzC[]
            C = CuSparseMatrixCSC(rowPtrC, cuzeros(Cint, nnz), cuzeros($elty, nnz), nnz, A.dims)
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint, Ptr{$elty},
                               Ptr{cusparseMatDescr_t}, Cint, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, Ptr{$elty},
                               Ptr{cusparseMatDescr_t}, Cint, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, Ptr{cusparseMatDescr_t},
                               CuPtr{$elty}, CuPtr{Cint}, CuPtr{Cint}),
                              handle(), mA, nA, [alpha], Ref(cudesca),
                              A.nnz, A.nzVal, A.colPtr, A.rowVal, [beta],
                              Ref(cudescb), B.nnz, B.nzVal, B.colPtr, B.rowVal,
                              Ref(cudescc), C.nzVal, C.colPtr, C.rowVal)
            C
        end
        function geam(alpha::$elty,
                      A::CuSparseMatrixCSR{$elty},
                      beta::$elty,
                      B::CuSparseMatrixCSC{$elty},
                      indexA::SparseChar,
                      indexB::SparseChar,
                      indexC::SparseChar)
            geam(alpha,A,beta,switch2csr(B),indexA,indexB,indexC)
        end
        function geam(alpha::$elty,
                      A::CuSparseMatrixCSC{$elty},
                      beta::$elty,
                      B::CuSparseMatrixCSR{$elty},
                      indexA::SparseChar,
                      indexB::SparseChar,
                      indexC::SparseChar)
            geam(alpha,switch2csr(A),beta,B,indexA,indexB,indexC)
        end
        function geam(alpha::$elty,
                      A::Union{CuSparseMatrixCSR{$elty},CuSparseMatrixCSC{$elty}},
                      B::Union{CuSparseMatrixCSR{$elty},CuSparseMatrixCSC{$elty}},
                      indexA::SparseChar,
                      indexB::SparseChar,
                      indexC::SparseChar)
            geam(alpha,A,one($elty),B,indexA,indexB,indexC)
        end
        function geam(A::Union{CuSparseMatrixCSR{$elty},CuSparseMatrixCSC{$elty}},
                      beta::$elty,
                      B::Union{CuSparseMatrixCSR{$elty},CuSparseMatrixCSC{$elty}},
                      indexA::SparseChar,
                      indexB::SparseChar,
                      indexC::SparseChar)
            geam(one($elty),A,beta,B,indexA,indexB,indexC)
        end
        function geam(A::Union{CuSparseMatrixCSR{$elty},CuSparseMatrixCSC{$elty}},
                      B::Union{CuSparseMatrixCSR{$elty},CuSparseMatrixCSC{$elty}},
                      indexA::SparseChar,
                      indexB::SparseChar,
                      indexC::SparseChar)
            geam(one($elty),A,one($elty),B,indexA,indexB,indexC)
        end
    end
end

"""
    gemm(transa::SparseChar, transb::SparseChar, A::CuSparseMatrix, B::CuSparseMatrix, indexA::SparseChar, indexB::SparseChar, indexC::SparseChar)

Solves `C = op(A)*op(B)`. `op(A)` can be nothing (`transa = N`), transpose 
(`transa = T`), or conjugate transpose (`transa = C`), and similarly for `op(B)` and
`transb`. All of `A`, `B`, and `C` are sparse.
"""
function gemm(transa::SparseChar, transb::SparseChar, A::CuSparseMatrix, B::CuSparseMatrix, indexA::SparseChar, indexB::SparseChar, indexC::SparseChar) end
for (fname,elty) in ((:cusparseScsrgemm, :Float32),
                     (:cusparseDcsrgemm, :Float64),
                     (:cusparseCcsrgemm, :ComplexF32),
                     (:cusparseZcsrgemm, :ComplexF64))
    @eval begin
        function gemm(transa::SparseChar,
                      transb::SparseChar,
                      A::CuSparseMatrixCSR{$elty},
                      B::CuSparseMatrixCSR{$elty},
                      indexA::SparseChar,
                      indexB::SparseChar,
                      indexC::SparseChar)
            cutransa = cusparseop(transa)
            cutransb = cusparseop(transb)
            cuinda   = cusparseindex(indexA)
            cuindb   = cusparseindex(indexB)
            cuindc   = cusparseindex(indexB)
            cudesca  = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            cudescb  = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindb)
            cudescc  = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindc)
            m,k  = transa == 'N' ? A.dims : (A.dims[2],A.dims[1])
            kB,n = transb == 'N' ? B.dims : (B.dims[2],B.dims[1])
            if k != kB
                throw(DimensionMismatch("Interior dimension of A, $k, and B, $kB, must match"))
            end
            nnzC = Ref{Cint}(1)
            rowPtrC = CuArray(zeros(Cint,m + 1))
            @check ccall((:cusparseXcsrgemmNnz,libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t,
                               cusparseOperation_t, Cint, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Cint, CuPtr{Cint},
                               CuPtr{Cint}, Ptr{cusparseMatDescr_t}, Cint, CuPtr{Cint},
                               CuPtr{Cint}, Ptr{cusparseMatDescr_t}, CuPtr{Cint},
                               Ptr{Cint}), handle(), cutransa, cutransb,
                               m, n, k, Ref(cudesca), A.nnz, A.rowPtr, A.colVal,
                               Ref(cudescb), B.nnz, B.rowPtr, B.colVal, Ref(cudescc),
                               rowPtrC, nnzC)
            nnz = nnzC[]
            C = CuSparseMatrixCSR(rowPtrC, CuArray(zeros(Cint,nnz)), CuArray(zeros($elty,nnz)), nnz, (m,n))
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t,
                               cusparseOperation_t, Cint, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Cint, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, Ptr{cusparseMatDescr_t},
                               Cint, CuPtr{$elty}, CuPtr{Cint}, CuPtr{Cint},
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}), handle(), cutransa,
                               cutransb, m, n, k, Ref(cudesca), A.nnz, A.nzVal,
                               A.rowPtr, A.colVal, Ref(cudescb), B.nnz, B.nzVal,
                               B.rowPtr, B.colVal, Ref(cudescc), C.nzVal,
                               C.rowPtr, C.colVal)
            C
        end
    end
end

#CSC GEMM
for (fname,elty) in ((:cusparseScsrgemm, :Float32),
                     (:cusparseDcsrgemm, :Float64),
                     (:cusparseCcsrgemm, :ComplexF32),
                     (:cusparseZcsrgemm, :ComplexF64))
    @eval begin
        function gemm(transa::SparseChar,
                      transb::SparseChar,
                      A::CuSparseMatrixCSC{$elty},
                      B::CuSparseMatrixCSC{$elty},
                      indexA::SparseChar,
                      indexB::SparseChar,
                      indexC::SparseChar)
            ctransa = 'N'
            if transa == 'N'
                ctransa = 'T'
            end
            cutransa = cusparseop(ctransa)
            ctransb = 'N'
            if transb == 'N'
                ctransb = 'T'
            end
            cutransb = cusparseop(ctransb)
            cuinda   = cusparseindex(indexA)
            cuindb   = cusparseindex(indexB)
            cuindc   = cusparseindex(indexB)
            cudesca  = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            cudescb  = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindb)
            cudescc  = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuindc)
            m,k  = ctransa != 'N' ? A.dims : (A.dims[2],A.dims[1])
            kB,n = ctransb != 'N' ? B.dims : (B.dims[2],B.dims[1])
            if k != kB
                throw(DimensionMismatch("Interior dimension of A, $k, and B, $kB, must match"))
            end
            nnzC = Ref{Cint}(1)
            colPtrC = CuArray(zeros(Cint,n + 1))
            @check ccall((:cusparseXcsrgemmNnz,libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t,
                               cusparseOperation_t, Cint, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Cint, CuPtr{Cint},
                               CuPtr{Cint}, Ptr{cusparseMatDescr_t}, Cint, CuPtr{Cint},
                               CuPtr{Cint}, Ptr{cusparseMatDescr_t}, CuPtr{Cint},
                               Ptr{Cint}), handle(), cutransa, cutransb,
                               m, n, k, Ref(cudesca), A.nnz, A.colPtr, A.rowVal,
                               Ref(cudescb), B.nnz, B.colPtr, B.rowVal, Ref(cudescc),
                               colPtrC, nnzC)
            nnz = nnzC[]
            C = CuSparseMatrixCSC(colPtrC, CuArray(zeros(Cint,nnz)), CuArray(zeros($elty,nnz)), nnz, (m,n))
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t,
                               cusparseOperation_t, Cint, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Cint, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, Ptr{cusparseMatDescr_t},
                               Cint, CuPtr{$elty}, CuPtr{Cint}, CuPtr{Cint},
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}), handle(), cutransa,
                               cutransb, m, n, k, Ref(cudesca), A.nnz, A.nzVal,
                               A.colPtr, A.rowVal, Ref(cudescb), B.nnz, B.nzVal,
                               B.colPtr, B.rowVal, Ref(cudescc), C.nzVal,
                               C.colPtr, C.rowVal)
            C
        end
    end
end

## preconditioners

"""
    ic0!(transa::SparseChar, typea::SparseChar, A::CompressedSparse, info::cusparseSolveAnalysisInfo_t, index::SparseChar)

Incomplete Cholesky factorization with no pivoting.
Preserves the sparse layout of matrix `A`. Must call
[`sv_analysis`](@ref) first, since this provides the `info` argument.
"""
function ic0!(transa::SparseChar, typea::SparseChar, A::CompressedSparse, info::cusparseSolveAnalysisInfo_t, index::SparseChar) end

for (fname,elty) in ((:cusparseScsric0, :Float32),
                     (:cusparseDcsric0, :Float64),
                     (:cusparseCcsric0, :ComplexF32),
                     (:cusparseZcsric0, :ComplexF64))
    @eval begin
        function ic0!(transa::SparseChar,
                      typea::SparseChar,
                      A::CompressedSparse{$elty},
                      info::cusparseSolveAnalysisInfo_t,
                      index::SparseChar)
            Mat     = A
            if typeof(A) <: HermOrSym
                Mat = A.data
            end
            cutransa = cusparseop(transa)
            cutype   = cusparsetype(typea)
            if typeof(A) <: Symmetric
                cutype = cusparsetype('S')
            elseif typeof(A) <: Hermitian
                cutype = cusparsetype('H')
            end

            if transa == 'N' && typeof(Mat) == CuSparseMatrixCSC{$elty}
                cutransa = cusparseop('T')
            elseif transa == 'T' && typeof(Mat) == CuSparseMatrixCSC{$elty}
                cutransa = cusparseop('N')
            end
            cuind    = cusparseindex(index)
            cudesc   = getDescr(A, index)
            m,n      = Mat.dims
            indPtr   = typeof(Mat) == CuSparseMatrixCSC{$elty} ? Mat.colPtr : Mat.rowPtr
            valPtr   = typeof(Mat) == CuSparseMatrixCSC{$elty} ? Mat.rowVal : Mat.colVal
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, cusparseSolveAnalysisInfo_t),
                              handle(), cutransa, m, Ref(cudesc), Mat.nzVal,
                              indPtr, valPtr, info)
            Mat
        end
        function ic0(transa::SparseChar,
                     typea::SparseChar,
                     A::CompressedSparse{$elty},
                     info::cusparseSolveAnalysisInfo_t,
                     index::SparseChar)
            ic0!(transa,typea,copy(A),info,index)
        end
    end
end

"""
    ic02!(A::CuSparseMatrix, index::SparseChar)

Incomplete Cholesky factorization with no pivoting.
Preserves the sparse layout of matrix `A`.
"""
function ic02!(A::CuSparseMatrix, index::SparseChar) end
for (bname,aname,sname,elty) in ((:cusparseScsric02_bufferSize, :cusparseScsric02_analysis, :cusparseScsric02, :Float32),
                                 (:cusparseDcsric02_bufferSize, :cusparseDcsric02_analysis, :cusparseDcsric02, :Float64),
                                 (:cusparseCcsric02_bufferSize, :cusparseCcsric02_analysis, :cusparseCcsric02, :ComplexF32),
                                 (:cusparseZcsric02_bufferSize, :cusparseZcsric02_analysis, :cusparseZcsric02, :ComplexF64))
    @eval begin
        function ic02!(A::CuSparseMatrixCSR{$elty},
                       index::SparseChar)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = csric02Info_t[0]
            cusparseCreateCsric02Info(info)
            bufSize = Ref{Cint}(1)
            @check ccall(($(string(bname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, csric02Info_t, Ptr{Cint}),
                              handle(), m, A.nnz, Ref(cudesc), A.nzVal,
                              A.rowPtr, A.colVal, info[1], bufSize)
            buffer = CuArray(zeros(UInt8, bufSize[]))
            @check ccall(($(string(aname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, csric02Info_t, cusparseSolvePolicy_t,
                               CuPtr{Cvoid}), handle(), m, A.nnz, Ref(cudesc),
                               A.nzVal, A.rowPtr, A.colVal, info[1],
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            posit = Ref{Cint}(1)
            @check ccall((:cusparseXcsric02_zeroPivot, libcusparse),
                        cusparseStatus_t, (cusparseHandle_t, csric02Info_t,
                        Ptr{Cint}), handle(), info[1], posit)
            if( posit[] >= 0 )
                throw(string("Structural/numerical zero in A at (",posit[],posit[],")"))
            end
            @check ccall(($(string(sname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, csric02Info_t, cusparseSolvePolicy_t,
                               CuPtr{Cvoid}), handle(), m, A.nnz,
                               Ref(cudesc), A.nzVal, A.rowPtr, A.colVal, info[1],
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
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
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = csric02Info_t[0]
            cusparseCreateCsric02Info(info)
            bufSize = Ref{Cint}(1)
            @check ccall(($(string(bname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, csric02Info_t, Ptr{Cint}),
                              handle(), m, A.nnz, Ref(cudesc), A.nzVal,
                              A.colPtr, A.rowVal, info[1], bufSize)
            buffer = CuArray(zeros(UInt8, bufSize[]))
            @check ccall(($(string(aname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, csric02Info_t, cusparseSolvePolicy_t,
                               CuPtr{Cvoid}), handle(), m, A.nnz, Ref(cudesc),
                               A.nzVal, A.colPtr, A.rowVal, info[1],
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            posit = Ref{Cint}(1)
            @check ccall((:cusparseXcsric02_zeroPivot, libcusparse),
                        cusparseStatus_t, (cusparseHandle_t, csric02Info_t,
                        Ptr{Cint}), handle(), info[1], posit)
            if( posit[] >= 0 )
                throw(string("Structural/numerical zero in A at (",posit[],posit[],")"))
            end
            @check ccall(($(string(sname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, csric02Info_t, cusparseSolvePolicy_t,
                               CuPtr{Cvoid}), handle(), m, A.nnz,
                               Ref(cudesc), A.nzVal, A.colPtr, A.rowVal, info[1],
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            cusparseDestroyCsric02Info(info[1])
            A
        end
    end
end

"""
    ilu0!(transa::SparseChar, A::CuSparseMatrix, info::cusparseSolveAnalysisInfo_t, index::SparseChar)

Incomplete LU factorization with no pivoting.
Preserves the sparse layout of matrix `A`. Must call
[`sv_analysis`](@ref) first, since this provides the `info` argument.
"""
function ilu0!(transa::SparseChar, A::CuSparseMatrix, info::cusparseSolveAnalysisInfo_t, index::SparseChar) end
for (fname,elty) in ((:cusparseScsrilu0, :Float32),
                     (:cusparseDcsrilu0, :Float64),
                     (:cusparseCcsrilu0, :ComplexF32),
                     (:cusparseZcsrilu0, :ComplexF64))
    @eval begin
        function ilu0!(transa::SparseChar,
                       A::CompressedSparse{$elty},
                       info::cusparseSolveAnalysisInfo_t,
                       index::SparseChar)
            Mat = A
            if typeof(A) <: HermOrSym
                Mat = A.data
            end
            cutransa = cusparseop(transa)
            if transa == 'N' && typeof(Mat) == CuSparseMatrixCSC{$elty}
                cutransa = cusparseop('T')
            elseif transa == 'T' && typeof(Mat) == CuSparseMatrixCSC{$elty}
                cutransa = cusparseop('N')
            end
            cuind    = cusparseindex(index)
            cudesc   = getDescr(A, index)
            m,n      = Mat.dims
            indPtr   = typeof(Mat) == CuSparseMatrixCSC{$elty} ? Mat.colPtr : Mat.rowPtr
            valPtr   = typeof(Mat) == CuSparseMatrixCSC{$elty} ? Mat.rowVal : Mat.colVal
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseOperation_t, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, cusparseSolveAnalysisInfo_t),
                              handle(), cutransa, m, Ref(cudesc), Mat.nzVal,
                              indPtr, valPtr, info)
            Mat
        end
        function ilu0(transa::SparseChar,
                      A::CompressedSparse{$elty},
                      info::cusparseSolveAnalysisInfo_t,
                      index::SparseChar)
            ilu0!(transa,copy(A),info,index)
        end
    end
end

"""
    ilu02!(A::CuSparseMatrix, index::SparseChar)

Incomplete LU factorization with no pivoting.
Preserves the sparse layout of matrix `A`.
"""
function ilu02!(A::CuSparseMatrix, index::SparseChar) end
for (bname,aname,sname,elty) in ((:cusparseScsrilu02_bufferSize, :cusparseScsrilu02_analysis, :cusparseScsrilu02, :Float32),
                                 (:cusparseDcsrilu02_bufferSize, :cusparseDcsrilu02_analysis, :cusparseDcsrilu02, :Float64),
                                 (:cusparseCcsrilu02_bufferSize, :cusparseCcsrilu02_analysis, :cusparseCcsrilu02, :ComplexF32),
                                 (:cusparseZcsrilu02_bufferSize, :cusparseZcsrilu02_analysis, :cusparseZcsrilu02, :ComplexF64))
    @eval begin
        function ilu02!(A::CuSparseMatrixCSR{$elty},
                        index::SparseChar)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = csrilu02Info_t[0]
            cusparseCreateCsrilu02Info(info)
            bufSize = Ref{Cint}(1)
            @check ccall(($(string(bname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, csrilu02Info_t, Ptr{Cint}),
                              handle(), m, A.nnz, Ref(cudesc), A.nzVal,
                              A.rowPtr, A.colVal, info[1], bufSize)
            buffer = CuArray(zeros(UInt8, bufSize[]))
            @check ccall(($(string(aname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, csrilu02Info_t, cusparseSolvePolicy_t,
                               CuPtr{Cvoid}), handle(), m, A.nnz, Ref(cudesc),
                               A.nzVal, A.rowPtr, A.colVal, info[1],
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            posit = Ref{Cint}(1)
            @check ccall((:cusparseXcsrilu02_zeroPivot, libcusparse),
                        cusparseStatus_t, (cusparseHandle_t, csrilu02Info_t,
                        Ptr{Cint}), handle(), info[1], posit)
            if( posit[] >= 0 )
                throw(string("Structural zero in A at (",posit[],posit[],")"))
            end
            @check ccall(($(string(sname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, csrilu02Info_t, cusparseSolvePolicy_t,
                               CuPtr{Cvoid}), handle(), m, A.nnz,
                               Ref(cudesc), A.nzVal, A.rowPtr, A.colVal, info[1],
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
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
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = csrilu02Info_t[0]
            cusparseCreateCsrilu02Info(info)
            bufSize = Ref{Cint}(1)
            @check ccall(($(string(bname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, csrilu02Info_t, Ptr{Cint}),
                              handle(), m, A.nnz, Ref(cudesc), A.nzVal,
                              A.colPtr, A.rowVal, info[1], bufSize)
            buffer = CuArray(zeros(UInt8, bufSize[]))
            @check ccall(($(string(aname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, csrilu02Info_t, cusparseSolvePolicy_t,
                               CuPtr{Cvoid}), handle(), m, A.nnz, Ref(cudesc),
                               A.nzVal, A.colPtr, A.rowVal, info[1],
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            posit = Ref{Cint}(1)
            @check ccall((:cusparseXcsrilu02_zeroPivot, libcusparse),
                        cusparseStatus_t, (cusparseHandle_t, csrilu02Info_t,
                        Ptr{Cint}), handle(), info[1], posit)
            if( posit[] >= 0 )
                throw(string("Structural zero in A at (",posit[],posit[],")"))
            end
            @check ccall(($(string(sname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, CuPtr{$elty}, CuPtr{Cint},
                               CuPtr{Cint}, csrilu02Info_t, cusparseSolvePolicy_t,
                               CuPtr{Cvoid}), handle(), m, A.nnz,
                               Ref(cudesc), A.nzVal, A.colPtr, A.rowVal, info[1],
                               CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
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
            cudir = cusparsedir(A.dir)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mb = div(m,A.blockDim)
            info = bsric02Info_t[0]
            cusparseCreateBsric02Info(info)
            bufSize = Ref{Cint}(1)
            @check ccall(($(string(bname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, Cint, bsric02Info_t,
                               Ptr{Cint}), handle(), cudir, mb, A.nnz,
                               Ref(cudesc), A.nzVal, A.rowPtr, A.colVal,
                               A.blockDim, info[1], bufSize)
            buffer = CuArray(zeros(UInt8, bufSize[]))
            @check ccall(($(string(aname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, Cint, bsric02Info_t,
                               cusparseSolvePolicy_t, CuPtr{Cvoid}),
                              handle(), cudir, mb, A.nnz, Ref(cudesc),
                              A.nzVal, A.rowPtr, A.colVal, A.blockDim, info[1],
                              CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            posit = Ref{Cint}(1)
            @check ccall((:cusparseXbsric02_zeroPivot, libcusparse),
                        cusparseStatus_t, (cusparseHandle_t, bsric02Info_t,
                        Ptr{Cint}), handle(), info[1], posit)
            if( posit[] >= 0 )
                throw(string("Structural/numerical zero in A at (",posit[],posit[],")"))
            end
            @check ccall(($(string(sname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, Cint,bsric02Info_t,
                               cusparseSolvePolicy_t, CuPtr{Cvoid}),
                              handle(), cudir, mb, A.nnz, Ref(cudesc),
                              A.nzVal, A.rowPtr, A.colVal, A.blockDim, info[1],
                              CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
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
            cudir = cusparsedir(A.dir)
            cuind = cusparseindex(index)
            cudesc = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuind)
            m,n = A.dims
            if( m != n )
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mb = div(m,A.blockDim)
            info = bsrilu02Info_t[0]
            cusparseCreateBsrilu02Info(info)
            bufSize = Ref{Cint}(1)
            @check ccall(($(string(bname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, Cint, bsrilu02Info_t,
                               Ptr{Cint}), handle(), cudir, mb, A.nnz,
                               Ref(cudesc), A.nzVal, A.rowPtr, A.colVal,
                               A.blockDim, info[1], bufSize)
            buffer = CuArray(zeros(UInt8, bufSize[]))
            @check ccall(($(string(aname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, Cint, bsrilu02Info_t,
                               cusparseSolvePolicy_t, CuPtr{Cvoid}),
                              handle(), cudir, mb, A.nnz, Ref(cudesc),
                              A.nzVal, A.rowPtr, A.colVal, A.blockDim, info[1],
                              CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            posit = Ref{Cint}(1)
            @check ccall((:cusparseXbsrilu02_zeroPivot, libcusparse),
                        cusparseStatus_t, (cusparseHandle_t, bsrilu02Info_t,
                        Ptr{Cint}), handle(), info[1], posit)
            if( posit[] >= 0 )
                throw(string("Structural/numerical zero in A at (",posit[],posit[],")"))
            end
            @check ccall(($(string(sname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, cusparseDirection_t, Cint,
                               Cint, Ptr{cusparseMatDescr_t}, CuPtr{$elty},
                               CuPtr{Cint}, CuPtr{Cint}, Cint,bsrilu02Info_t,
                               cusparseSolvePolicy_t, CuPtr{Cvoid}),
                              handle(), cudir, mb, A.nnz, Ref(cudesc),
                              A.nzVal, A.rowPtr, A.colVal, A.blockDim, info[1],
                              CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
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

"""
    gtsv!(dl::CuVector, d::CuVector, du::CuVector, B::CuMatrix)

Performs the solution of `A \\ B` where `A` is a tridiagonal matrix, with
lower diagonal `dl`, main diagonal `d`, and upper diagonal `du`.
"""
function gtsv!(dl::CuVector, d::CuVector, du::CuVector, B::CuMatrix) end

for (fname,elty) in ((:cusparseSgtsv, :Float32),
                     (:cusparseDgtsv, :Float64),
                     (:cusparseCgtsv, :ComplexF32),
                     (:cusparseZgtsv, :ComplexF64))
    @eval begin
        function gtsv!(dl::CuVector{$elty},
                       d::CuVector{$elty},
                       du::CuVector{$elty},
                       B::CuMatrix{$elty})
            m,n = B.dims
            ldb = max(1,stride(B,2))
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint, CuPtr{$elty},
                               CuPtr{$elty}, CuPtr{$elty}, CuPtr{$elty}, Cint),
                              handle(), m, n, dl, d, du, B, ldb)
            B
        end
        function gtsv(dl::CuVector{$elty},
                      d::CuVector{$elty},
                      du::CuVector{$elty},
                      B::CuMatrix{$elty})
            gtsv!(dl,d,du,copy(B))
        end
    end
end

"""
    gtsv_nopivot!(dl::CuVector, d::CuVector, du::CuVector, B::CuMatrix)

Performs the solution of `A \\ B` where `A` is a tridiagonal matrix, with
lower diagonal `dl`, main diagonal `d`, and upper diagonal `du`. No pivoting is used.
"""
function gtsv_nopivot!(dl::CuVector, d::CuVector, du::CuVector, B::CuMatrix) end
for (fname,elty) in ((:cusparseSgtsv_nopivot, :Float32),
                     (:cusparseDgtsv_nopivot, :Float64),
                     (:cusparseCgtsv_nopivot, :ComplexF32),
                     (:cusparseZgtsv_nopivot, :ComplexF64))
    @eval begin
        function gtsv_nopivot!(dl::CuVector{$elty},
                               d::CuVector{$elty},
                               du::CuVector{$elty},
                               B::CuMatrix{$elty})
            m,n = B.dims
            ldb = max(1,stride(B,2))
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, Cint, CuPtr{$elty},
                               CuPtr{$elty}, CuPtr{$elty}, CuPtr{$elty}, Cint),
                              handle(), m, n, dl, d, du, B, ldb)
            B
        end
        function gtsv_nopivot(dl::CuVector{$elty},
                              d::CuVector{$elty},
                              du::CuVector{$elty},
                              B::CuMatrix{$elty})
            gtsv_nopivot!(dl,d,du,copy(B))
        end
    end
end

"""
    gtsvStridedBatch!(dl::CuVector, d::CuVector, du::CuVector, X::CuVector, batchCount::Integer, batchStride::Integer)

Performs the batched solution of `A[i] \\ B[i]` where `A[i]` is a tridiagonal matrix, with
lower diagonal `dl`, main diagonal `d`, and upper diagonal `du`. `batchCount` determines
how many elements there are in the batch in total (how many `A`s?), and `batchStride` sets
the separation of each item in the batch (it must be at least `m`, the matrix dimension).
"""
function gtsvStridedBatch!(dl::CuVector, d::CuVector, du::CuVector, X::CuVector, batchCount::Integer, batchStride::Integer) end
for (fname,elty) in ((:cusparseSgtsvStridedBatch, :Float32),
                     (:cusparseDgtsvStridedBatch, :Float64),
                     (:cusparseCgtsvStridedBatch, :ComplexF32),
                     (:cusparseZgtsvStridedBatch, :ComplexF64))
    @eval begin
        function gtsvStridedBatch!(dl::CuVector{$elty},
                                   d::CuVector{$elty},
                                   du::CuVector{$elty},
                                   X::CuVector{$elty},
                                   batchCount::Integer,
                                   batchStride::Integer)
            m = div(length(X),batchCount)
            @check ccall(($(string(fname)),libcusparse), cusparseStatus_t,
                              (cusparseHandle_t, Cint, CuPtr{$elty}, CuPtr{$elty},
                               CuPtr{$elty}, CuPtr{$elty}, Cint, Cint),
                              handle(), m, dl, d, du, X,
                              batchCount, batchStride)
            X
        end
        function gtsvStridedBatch(dl::CuVector{$elty},
                                  d::CuVector{$elty},
                                  du::CuVector{$elty},
                                  X::CuVector{$elty},
                                  batchCount::Integer,
                                  batchStride::Integer)
            gtsvStridedBatch!(dl,d,du,copy(X),batchCount,batchStride)
        end
    end
end
