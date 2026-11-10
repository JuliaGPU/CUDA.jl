# routines that implement different preconditioners

export ic02!, ic02, ilu02!, ilu02, gtsv2!, gtsv2

"""
    ic02!(A; index='O') -> A

Compute the incomplete Cholesky factorization with zero fill-in, IC(0), of a sparse
Hermitian positive definite matrix in place, keeping the sparsity pattern of `A`. An error
is thrown when the factorization hits a zero pivot.

cuSPARSE reads the lower triangle of the matrix as stored row by row. For a CSR matrix, that
is the lower triangle of `A`, which is overwritten with the factor `L` of `A ≈ L * L'`. A
CSC matrix is processed as the CSR storage of its transpose: its upper triangle is read, and
overwritten with the factor `R = L'` of `A ≈ R' * R`. BSR matrices follow the conventions of
`bsric02`.

The result is meant as a preconditioner for iterative solvers, applied with `ldiv!` on
triangular views of it.

Wraps `cusparse<t>csric02` and `cusparse<t>bsric02`, which NVIDIA has deprecated, for the
`Float32`, `Float64`, `ComplexF32` and `ComplexF64` element types.
"""
function ic02! end

"""
    ilu02!(A; index='O') -> A

Compute the incomplete LU factorization with zero fill-in, ILU(0), of the sparse square
matrix `A` in place, keeping its sparsity pattern. `A` is overwritten with the factors `L`,
with an implicit unit diagonal, and `U` of `A ≈ L * U`. An error is thrown when the
factorization hits a zero pivot.

The result is meant as a preconditioner for iterative solvers, applied with `ldiv!` on
triangular views of it.

Wraps `cusparse<t>csrilu02` and `cusparse<t>bsrilu02`, which NVIDIA has deprecated, for the
`Float32`, `Float64`, `ComplexF32` and `ComplexF64` element types.
"""
function ilu02! end

"""
    gtsv2!(dl::CuVector, d::CuVector, du::CuVector, B::CuVecOrMat; pivoting=true) -> B

Solve `A * X = B` in place in `B`, for the tridiagonal matrix `A` with subdiagonal `dl`,
diagonal `d` and superdiagonal `du`. The three vectors have the same length as the columns
of `B`, at least 3, and `dl[1]` and `du[end]` must be zero. With `pivoting`, partial
pivoting makes the solution more accurate but slower.

Wraps `cusparse<t>gtsv2` and `cusparse<t>gtsv2_nopivot`.
"""
function gtsv2! end

# csric02
for (bname,aname,sname,elty) in ((:cusparseScsric02_bufferSize, :cusparseScsric02_analysis, :cusparseScsric02, :Float32),
                                 (:cusparseDcsric02_bufferSize, :cusparseDcsric02_analysis, :cusparseDcsric02, :Float64),
                                 (:cusparseCcsric02_bufferSize, :cusparseCcsric02_analysis, :cusparseCcsric02, :ComplexF32),
                                 (:cusparseZcsric02_bufferSize, :cusparseZcsric02_analysis, :cusparseZcsric02, :ComplexF64))
    @eval begin
        function ic02!(A::CuSparseMatrixCSR{$elty}; index::SparseChar='O')
            desc = CuMatrixDescriptor('G', 'L', 'N', index)
            m,n = size(A)
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = IC0Info()

            function bufferSize()
                out = Ref{Cint}(1)
                $bname(handle(), m, nnz(A), desc, nonzeros(A), A.rowPtr, A.colVal, info,
                       out)
                return out[]
            end
            with_workspace(bufferSize) do buffer
                $aname(handle(), m, nnz(A), desc,
                        nonzeros(A), A.rowPtr, A.colVal, info,
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                posit = Ref{Cint}(1)
                cusparseXcsric02_zeroPivot(handle(), info, posit)
                if posit[] >= 0
                    error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                end
                $sname(handle(), m, nnz(A),
                        desc, nonzeros(A), A.rowPtr, A.colVal, info,
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            end
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
        function ic02!(A::CuSparseMatrixCSC{$elty}; index::SparseChar='O')
            desc = CuMatrixDescriptor('G', 'L', 'N', index)
            m,n = size(A)
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = IC0Info()

            function bufferSize()
                out = Ref{Cint}(1)
                $bname(handle(), m, nnz(A), desc, nonzeros(A), A.colPtr, rowvals(A),
                       info, out)
                return out[]
            end
            with_workspace(bufferSize) do buffer
                $aname(handle(), m, nnz(A), desc,
                        nonzeros(A), A.colPtr, rowvals(A), info,
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                posit = Ref{Cint}(1)
                cusparseXcsric02_zeroPivot(handle(), info, posit)
                if posit[] >= 0
                    error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                end
                $sname(handle(), m, nnz(A),
                        desc, nonzeros(A), A.colPtr, rowvals(A), info,
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            end
            A
        end
    end
end

# csrilu02
for (bname,aname,sname,elty) in ((:cusparseScsrilu02_bufferSize, :cusparseScsrilu02_analysis, :cusparseScsrilu02, :Float32),
                                 (:cusparseDcsrilu02_bufferSize, :cusparseDcsrilu02_analysis, :cusparseDcsrilu02, :Float64),
                                 (:cusparseCcsrilu02_bufferSize, :cusparseCcsrilu02_analysis, :cusparseCcsrilu02, :ComplexF32),
                                 (:cusparseZcsrilu02_bufferSize, :cusparseZcsrilu02_analysis, :cusparseZcsrilu02, :ComplexF64))
    @eval begin
        function ilu02!(A::CuSparseMatrixCSR{$elty}; index::SparseChar='O')
            desc = CuMatrixDescriptor('G', 'L', 'N', index)
            m,n = size(A)
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = ILU0Info()

            function bufferSize()
                out = Ref{Cint}(1)
                $bname(handle(), m, nnz(A), desc,
                       nonzeros(A), A.rowPtr, A.colVal, info,
                       out)
                return out[]
            end
            with_workspace(bufferSize) do buffer
                $aname(handle(), m, nnz(A), desc,
                        nonzeros(A), A.rowPtr, A.colVal, info,
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                posit = Ref{Cint}(1)
                cusparseXcsrilu02_zeroPivot(handle(), info, posit)
                if posit[] >= 0
                    error("Structural zero in A at ($(posit[]),$(posit[])))")
                end
                $sname(handle(), m, nnz(A),
                        desc, nonzeros(A), A.rowPtr, A.colVal, info,
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            end
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
        function ilu02!(A::CuSparseMatrixCSC{$elty}; index::SparseChar='O')
            desc = CuMatrixDescriptor('G', 'L', 'N', index)
            m,n = size(A)
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            info = ILU0Info()

            function bufferSize()
                out = Ref{Cint}(1)
                $bname(handle(), m, nnz(A), desc,
                       nonzeros(A), A.colPtr, rowvals(A), info,
                       out)
                return out[]
            end
            with_workspace(bufferSize) do buffer
                $aname(handle(), m, nnz(A), desc,
                        nonzeros(A), A.colPtr, rowvals(A), info,
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                posit = Ref{Cint}(1)
                cusparseXcsrilu02_zeroPivot(handle(), info, posit)
                if posit[] >= 0
                    error("Structural zero in A at ($(posit[]),$(posit[])))")
                end
                $sname(handle(), m, nnz(A),
                        desc, nonzeros(A), A.colPtr, rowvals(A), info,
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            end
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
        function ic02!(A::CuSparseMatrixBSR{$elty}; index::SparseChar='O')
            desc = CuMatrixDescriptor('G', 'U', 'N', index)
            m,n = size(A)
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mb = div(m,A.blockDim)
            info = IC0InfoBSR()

            function bufferSize()
                out = Ref{Cint}(1)
                $bname(handle(), A.dir, mb, nnz(A), desc, nonzeros(A),
                       A.rowPtr, A.colVal, A.blockDim, info, out)
                return out[]
            end
            with_workspace(bufferSize) do buffer
                $aname(handle(), A.dir, mb, nnz(A), desc,
                       nonzeros(A), A.rowPtr, A.colVal, A.blockDim, info,
                       CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                posit = Ref{Cint}(1)
                cusparseXbsric02_zeroPivot(handle(), info, posit)
                if posit[] >= 0
                    error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                end
                $sname(handle(), A.dir, mb, nnz(A), desc,
                       nonzeros(A), A.rowPtr, A.colVal, A.blockDim, info,
                       CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            end
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
        function ilu02!(A::CuSparseMatrixBSR{$elty}; index::SparseChar='O')
            desc = CuMatrixDescriptor('G', 'U', 'N', index)
            m,n = size(A)
            if m != n
                throw(DimensionMismatch("A must be square, but has dimensions ($m,$n)!"))
            end
            mb = div(m,A.blockDim)
            info = ILU0InfoBSR()

            function bufferSize()
                out = Ref{Cint}(1)
                $bname(handle(), A.dir, mb, nnz(A), desc, nonzeros(A),
                       A.rowPtr, A.colVal, A.blockDim, info, out)
                return out[]
            end
            with_workspace(bufferSize) do buffer
                $aname(handle(), A.dir, mb, nnz(A), desc,
                        nonzeros(A), A.rowPtr, A.colVal, A.blockDim, info,
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
                posit = Ref{Cint}(1)
                cusparseXbsrilu02_zeroPivot(handle(), info, posit)
                if posit[] >= 0
                    error("Structural/numerical zero in A at ($(posit[]),$(posit[])))")
                end
                $sname(handle(), A.dir, mb, nnz(A), desc,
                        nonzeros(A), A.rowPtr, A.colVal, A.blockDim, info,
                        CUSPARSE_SOLVE_POLICY_USE_LEVEL, buffer)
            end
            A
        end
    end
end

"""
    ic02(A; index='O')

Compute the incomplete Cholesky factorization IC(0) of the sparse matrix `A`, like
[`ic02!`](@ref) but in a copy of `A`.
"""
function ic02 end

"""
    ilu02(A; index='O')

Compute the incomplete LU factorization ILU(0) of the sparse matrix `A`, like
[`ilu02!`](@ref) but in a copy of `A`.
"""
function ilu02 end

for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        function ilu02(A::CuSparseMatrix{$elty}; index::SparseChar='O')
            isa(A, CuSparseMatrixCOO) && throw(ErrorException("ILU(0) decomposition of CuSparseMatrixCOO is not supported by the current CUDA version."))
            ilu02!(copy(A); index)
        end
        function ic02(A::CuSparseMatrix{$elty}; index::SparseChar='O')
            isa(A, CuSparseMatrixCOO) && throw(ErrorException("IC(0) decomposition of CuSparseMatrixCOO is not supported by the current CUDA version."))
            ic02!(copy(A); index)
        end
        function ilu02(A::HermOrSym{$elty,CuSparseMatrix{$elty}}; index::SparseChar='O')
            isa(A, CuSparseMatrixCOO) && throw(ErrorException("ILU(0) decomposition of CuSparseMatrixCOO is not supported by the current CUDA version."))
            ilu02!(copy(A.data); index)
        end
        function ic02(A::HermOrSym{$elty,CuSparseMatrix{$elty}}; index::SparseChar='O')
            isa(A, CuSparseMatrixCOO) && throw(ErrorException("IC(0) decomposition of CuSparseMatrixCOO is not supported by the current CUDA version."))
            ic02!(copy(A.data); index)
        end
    end
end

# gtsv2
for (bname_pivot,fname_pivot,bname_nopivot,fname_nopivot,elty) in ((:cusparseSgtsv2_bufferSizeExt, :cusparseSgtsv2, :cusparseSgtsv2_nopivot_bufferSizeExt, :cusparseSgtsv2_nopivot, :Float32),
                                                                   (:cusparseDgtsv2_bufferSizeExt, :cusparseDgtsv2, :cusparseDgtsv2_nopivot_bufferSizeExt, :cusparseDgtsv2_nopivot, :Float64),
                                                                   (:cusparseCgtsv2_bufferSizeExt, :cusparseCgtsv2, :cusparseCgtsv2_nopivot_bufferSizeExt, :cusparseCgtsv2_nopivot, :ComplexF32),
                                                                   (:cusparseZgtsv2_bufferSizeExt, :cusparseZgtsv2, :cusparseZgtsv2_nopivot_bufferSizeExt, :cusparseZgtsv2_nopivot, :ComplexF64))
    @eval begin
        function gtsv2!(dl::CuVector{$elty}, d::CuVector{$elty}, du::CuVector{$elty}, B::CuVecOrMat{$elty}; pivoting::Bool=true)
            ml = length(dl)
            m = length(d)
            mu = length(du)
            mB = size(B,1)
            (m ≤ 2) && throw(DimensionMismatch("The size of the linear system must be at least 3."))
            !(ml == m == mu) && throw(DimensionMismatch("(dl, d, du) must have the same length, the size of the vectors is ($ml,$m,$mu)!"))
            (m != mB) && throw(DimensionMismatch("The tridiagonal matrix and the right-hand side B have inconsistent dimensions ($m != $mB)!"))
            n = size(B,2)
            ldb = max(1,stride(B,2))

            function bufferSize()
                out = Ref{Csize_t}(1)
                if pivoting
                    $bname_pivot(handle(), m, n, dl, d, du, B, ldb, out)
                else
                    $bname_nopivot(handle(), m, n, dl, d, du, B, ldb, out)
                end
                return out[]
            end
            with_workspace(bufferSize) do buffer
                if pivoting
                    $fname_pivot(handle(), m, n, dl, d, du, B, ldb, buffer)
                else
                    $fname_nopivot(handle(), m, n, dl, d, du, B, ldb, buffer)
                end
            end
            B
        end
    end
end

"""
    gtsv2(dl::CuVector, d::CuVector, du::CuVector, B::CuVecOrMat; pivoting=true) -> X

Solve `A * X = B` for the tridiagonal matrix `A`, like [`gtsv2!`](@ref) but without
overwriting `B`.
"""
function gtsv2 end

for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        function gtsv2(dl::CuVector{$elty}, d::CuVector{$elty}, du::CuVector{$elty}, B::CuVecOrMat{$elty}; pivoting::Bool=true)
            gtsv2!(dl, d, du, copy(B); pivoting)
        end
    end
end
