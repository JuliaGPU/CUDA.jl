# routines that implement different preconditioners

export ic02!, ic02, ilu02!, ilu02, gtsv2!, gtsv2

"""
    ic02!(A::CuSparseMatrix, index::SparseChar='O')

Incomplete Cholesky factorization with no pivoting.
Preserves the sparse layout of matrix `A`.
"""
function ic02! end

"""
    ilu02!(A::CuSparseMatrix, index::SparseChar='O')

Incomplete LU factorization with no pivoting.
Preserves the sparse layout of matrix `A`.
"""
function ilu02! end

"""
    gtsv2!(dl::CuVector, d::CuVector, du::CuVector, B::CuVecOrMat, index::SparseChar='O'; pivoting::Bool=true)

Solve the linear system `A * X = B` where `A` is a tridiagonal matrix defined
by three vectors corresponding to its lower (`dl`), main (`d`), and upper (`du`) diagonals.
With `pivoting`, the solution is more accurate but also more expensive.
Note that the solution `X` overwrites the right-hand side `B`.
"""
function gtsv2! end

# csric02
for (bname,aname,sname,elty) in ((:cusparseScsric02_bufferSize, :cusparseScsric02_analysis, :cusparseScsric02, :Float32),
                                 (:cusparseDcsric02_bufferSize, :cusparseDcsric02_analysis, :cusparseDcsric02, :Float64),
                                 (:cusparseCcsric02_bufferSize, :cusparseCcsric02_analysis, :cusparseCcsric02, :ComplexF32),
                                 (:cusparseZcsric02_bufferSize, :cusparseZcsric02_analysis, :cusparseZcsric02, :ComplexF64))
    @eval begin
        function ic02!(A::CuSparseMatrixCSR{$elty}, index::SparseChar='O')
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
        function ic02!(A::CuSparseMatrixCSC{$elty}, index::SparseChar='O')
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
        function ilu02!(A::CuSparseMatrixCSR{$elty}, index::SparseChar='O')
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
        function ilu02!(A::CuSparseMatrixCSC{$elty}, index::SparseChar='O')
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
        function ic02!(A::CuSparseMatrixBSR{$elty}, index::SparseChar='O')
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
        function ilu02!(A::CuSparseMatrixBSR{$elty}, index::SparseChar='O')
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

for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        function ilu02(A::CuSparseMatrix{$elty}, index::SparseChar='O')
            isa(A, CuSparseMatrixCOO) && throw(ErrorException("ILU(0) decomposition of CuSparseMatrixCOO is not supported by the current CUDA version."))
            ilu02!(copy(A),index)
        end
        function ic02(A::CuSparseMatrix{$elty}, index::SparseChar='O')
            isa(A, CuSparseMatrixCOO) && throw(ErrorException("IC(0) decomposition of CuSparseMatrixCOO is not supported by the current CUDA version."))
            ic02!(copy(A),index)
        end
        function ilu02(A::HermOrSym{$elty,CuSparseMatrix{$elty}}, index::SparseChar='O')
            isa(A, CuSparseMatrixCOO) && throw(ErrorException("ILU(0) decomposition of CuSparseMatrixCOO is not supported by the current CUDA version."))
            ilu02!(copy(A.data),index)
        end
        function ic02(A::HermOrSym{$elty,CuSparseMatrix{$elty}}, index::SparseChar='O')
            isa(A, CuSparseMatrixCOO) && throw(ErrorException("IC(0) decomposition of CuSparseMatrixCOO is not supported by the current CUDA version."))
            ic02!(copy(A.data),index)
        end
    end
end

# gtsv2
for (bname_pivot,fname_pivot,bname_nopivot,fname_nopivot,elty) in ((:cusparseSgtsv2_bufferSizeExt, :cusparseSgtsv2, :cusparseSgtsv2_nopivot_bufferSizeExt, :cusparseSgtsv2_nopivot, :Float32),
                                                                   (:cusparseDgtsv2_bufferSizeExt, :cusparseDgtsv2, :cusparseDgtsv2_nopivot_bufferSizeExt, :cusparseDgtsv2_nopivot, :Float64),
                                                                   (:cusparseCgtsv2_bufferSizeExt, :cusparseCgtsv2, :cusparseCgtsv2_nopivot_bufferSizeExt, :cusparseCgtsv2_nopivot, :ComplexF32),
                                                                   (:cusparseZgtsv2_bufferSizeExt, :cusparseZgtsv2, :cusparseZgtsv2_nopivot_bufferSizeExt, :cusparseZgtsv2_nopivot, :ComplexF64))
    @eval begin
        function gtsv2!(dl::CuVector{$elty}, d::CuVector{$elty}, du::CuVector{$elty}, B::CuVecOrMat{$elty}, index::SparseChar='O'; pivoting::Bool=true)
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

for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        function gtsv2(dl::CuVector{$elty}, d::CuVector{$elty}, du::CuVector{$elty}, B::CuVecOrMat{$elty}, index::SparseChar='O'; pivoting::Bool=true)
            gtsv2!(dl, d, du, copy(B), index; pivoting)
        end
    end
end
