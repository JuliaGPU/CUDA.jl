# sparse linear algebra functions that perform operations between sparse and (usually tall)
# dense matrices

export sm2!, sm2

"""
    sm2!(transa, transxy, uplo, diag, alpha, A::CuSparseMatrixBSR, X::StridedCuMatrix; index='O') -> X

Solve `op(A) * op(Y) = alpha * op(X)` in place in `X`, for a sparse triangular matrix `A`
in BSR format. `transa` selects `op` for `A` (`'N'`, `'T'` or `'C'`), and `transxy` for `X`
and `Y` (`'N'` or `'T'`, and `'C'` for real matrices only).

`uplo` selects the triangle of `A` to use (`'U'` or `'L'`), and `diag` is `'U'` for a unit
diagonal, which is not read, or `'N'`. An error is thrown when the diagonal has a structural
or numerical zero. For CSR, CSC and COO matrices, use [`sm!`](@ref).

Wraps `cusparse<t>bsrsm2_solve`, which NVIDIA has deprecated, for the `Float32`, `Float64`,
`ComplexF32` and `ComplexF64` element types.
"""
function sm2! end

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
                      X::StridedCuMatrix{$elty};
                      index::SparseChar='O')

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

"""
    sm2(transa, transxy, uplo, [diag='N'], [alpha=1], A::CuSparseMatrixBSR, X::StridedCuMatrix; index='O') -> Y

Solve `op(A) * op(Y) = alpha * op(X)` for `Y`, like [`sm2!`](@ref) but without overwriting `X`.
`alpha` can only be given together with `diag`.
"""
function sm2(transa::SparseChar, transxy::SparseChar, uplo::SparseChar, diag::SparseChar,
             alpha::Number, A::CuSparseMatrixBSR{T}, X::StridedCuMatrix{T};
             index::SparseChar='O') where T
    sm2!(transa,transxy,uplo,diag,alpha,A,copy(X); index)
end
function sm2(transa::SparseChar, transxy::SparseChar, uplo::SparseChar, diag::SparseChar,
              A::CuSparseMatrixBSR{T}, X::StridedCuMatrix{T}; index::SparseChar='O') where T
    sm2!(transa,transxy,uplo,diag,one(T),A,copy(X); index)
end
function sm2(transa::SparseChar, transxy::SparseChar, uplo::SparseChar,
             A::CuSparseMatrixBSR{T}, X::StridedCuMatrix{T}; index::SparseChar='O') where T
    sm2!(transa,transxy,uplo,'N',one(T),A,copy(X); index)
end
