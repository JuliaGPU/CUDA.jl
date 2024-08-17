mutable struct SparseQRInfo
    info::csrqrInfo_t

    function SparseQRInfo()
        info_ref = Ref{csrqrInfo_t}()
        cusolverSpCreateCsrqrInfo(info_ref)
        obj = new(info_ref[])
        finalizer(cusolverSpDestroyCsrqrInfo, obj)
        obj
    end
end

Base.unsafe_convert(::Type{csrqrInfo_t}, info::SparseQRInfo) = info.info

mutable struct SparseQR{T <: BlasFloat} <: Factorization{T}
  n::Cint
  m::Cint
  nnzA::Cint
  mu::T
  descA::CuMatrixDescriptor
  info::SparseQRInfo
  buffer::Union{CuPtr{Cvoid},CuVector{UInt8}}
end

function SparseQR(A::CuSparseMatrixCSR{T,Cint}, index::Char='O') where T <: BlasFloat
    m,n = size(A)
    nnzA = nnz(A)
    mu = zero(T)
    descA = CuMatrixDescriptor('G', 'L', 'N', index)
    info = SparseQRInfo()
    buffer = CU_NULL
    F = SparseQR{T}(n, m, nnzA, mu, descA, info, buffer)
    spqr_analyse(F, A)
    spqr_buffer(F, A)
    return F
end

# csrqrAnalysis
#
# cusolverStatus_t cusolverSpXcsrqrAnalysis(
# cusolverSpHandle_t       handle,
# int                      m,
# int                      n,
# int                      nnzA,
# const cusparseMatDescr_t descrA,
# const int *              csrRowPtrA,
# const int *              csrColIndA,
# csrqrInfo_t              info);
function spqr_analyse(F::SparseQR{T}, A::CuSparseMatrixCSR{T,Cint}) where T <: BlasFloat
    cusolverSpXcsrqrAnalysis(sparse_handle(), F.m, F.n, F.nnzA, F.descA, A.rowPtr, A.colVal, F.info)
    return F
end

for (bname, iname, fname, sname, pname, elty, relty) in
    ((:cusolverSpScsrqrBufferInfo, :cusolverSpScsrqrSetup, :cusolverSpScsrqrFactor, :cusolverSpScsrqrSolve, :cusolverSpScsrqrZeroPivot, :Float32   , :Float32),
     (:cusolverSpDcsrqrBufferInfo, :cusolverSpDcsrqrSetup, :cusolverSpDcsrqrFactor, :cusolverSpDcsrqrSolve, :cusolverSpDcsrqrZeroPivot, :Float64   , :Float64),
     (:cusolverSpCcsrqrBufferInfo, :cusolverSpCcsrqrSetup, :cusolverSpCcsrqrFactor, :cusolverSpCcsrqrSolve, :cusolverSpCcsrqrZeroPivot, :ComplexF32, :Float32),
     (:cusolverSpZcsrqrBufferInfo, :cusolverSpZcsrqrSetup, :cusolverSpZcsrqrFactor, :cusolverSpZcsrqrSolve, :cusolverSpZcsrqrZeroPivot, :ComplexF64, :Float64))
    @eval begin
        # csrqrBufferInfo
        #
        # cusolverStatus_t cusolverSpScsrqrBufferInfo(
        # cusolverSpHandle_t       handle,
        # int                      m,
        # int                      n,
        # int                      nnzA,
        # const cusparseMatDescr_t descrA,
        # const float *            csrValA,
        # const int *              csrRowPtrA,
        # const int *              csrColIndA,
        # csrqrInfo_t              info,
        # size_t *                 internalDataInBytes,
        # size_t *                 workspaceInBytes);
        function spqr_buffer(F::SparseQR{$elty}, A::CuSparseMatrixCSR{$elty,Cint})
            internalDataInBytes = Ref{Csize_t}(0)
            workspaceInBytes = Ref{Csize_t}(0)
            $bname(sparse_handle(), F.m, F.n, F.nnzA, F.descA, A.nzVal, A.rowPtr, A.colVal, F.info, internalDataInBytes, workspaceInBytes)
            F.buffer = CuVector{UInt8}(undef, workspaceInBytes[])
            return F
        end

        # csrqrSetup
        #
        # cusolverStatus_t cusolverSpDcsrqrSetup(
        # cusolverSpHandle_t       handle,
        # int                      m,
        # int                      n,
        # int                      nnzA,
        # const cusparseMatDescr_t descrA,
        # const double *           csrValA,
        # const int *              csrRowPtrA,
        # const int *              csrColIndA,
        # double                   mu,
        # csrqrInfo_t              info);
        #
        # csrqrFactor
        #
        # cusolverStatus_t cusolverSpDcsrqrFactor(
        # cusolverSpHandle_t handle,
        # int                m,
        # int                n,
        # int                nnzA,
        # double *           b,
        # double *           x,
        # csrqrInfo_t        info,
        # void *             pBuffer);
        #
        # csrqrZeroPivot
        #
        # cusolverStatus_t cusolverSpDcsrqrZeroPivot(
        # cusolverSpHandle_t handle,
        # csrqrInfo_t        info,
        # double             tol,
        # int *              position);
        function spqr_factorise(F::SparseQR{$elty}, A::CuSparseMatrixCSR{$elty,Cint}, tol::$relty)
            $iname(sparse_handle(), F.m, F.n, F.nnzA, F.descA, A.nzVal, A.rowPtr, A.colVal, F.mu, F.info)
            $fname(sparse_handle(), F.m, F.n, F.nnzA, CU_NULL, CU_NULL, F.info, F.buffer)
            singularity = Ref{Cint}(0)
            $pname(sparse_handle(), F.info, tol, singularity)
            (singularity[] ≥ 0) && throw(SingularException(singularity[]))
            return F
        end

        function spqr_factorise_solve(F::SparseQR{$elty}, A::CuSparseMatrixCSR{$elty,Cint}, b::CuVector{$elty}, x::CuVector{$elty}, tol::$relty)
            $iname(sparse_handle(), F.m, F.n, F.nnzA, F.descA, A.nzVal, A.rowPtr, A.colVal, F.mu, F.info)
            $fname(sparse_handle(), F.m, F.n, F.nnzA, b, x, F.info, F.buffer)
            singularity = Ref{Cint}(0)
            $pname(sparse_handle(), F.info, tol, singularity)
            (singularity[] ≥ 0) && throw(SingularException(singularity[]))
            return F
        end

        # csrqrSolve
        #
        # cusolverStatus_t CUSOLVERAPI cusolverSpScsrqrSolve(
        # cusolverSpHandle_t handle,
        # int                m,
        # int                n,
        # float *            b,
        # float *            x,
        # csrqrInfo_t        info,
        # void *             pBuffer);
        function spqr_solve(F::SparseQR{$elty}, b::CuVector{$elty}, x::CuVector{$elty})
            $sname(sparse_handle(), F.m, F.n, b, x, F.info, F.buffer)
            return x
        end

        function spqr_solve(F::SparseQR{$elty}, B::CuMatrix{$elty}, X::CuMatrix{$elty})
            m, p = size(B)
            for j=1:p
                $sname(sparse_handle(), F.m, F.n, view(B,:,j), view(X,:,j), F.info, F.buffer)
            end
            return X
        end
    end
end

mutable struct SparseCholeskyInfo
    info::csrcholInfo_t

    function SparseCholeskyInfo()
        info_ref = Ref{csrcholInfo_t}()
        cusolverSpCreateCsrcholInfo(info_ref)
        obj = new(info_ref[])
        finalizer(cusolverSpDestroyCsrcholInfo, obj)
        obj
    end
end

Base.unsafe_convert(::Type{csrcholInfo_t}, info::SparseCholeskyInfo) = info.info

mutable struct SparseCholesky{T <: BlasFloat} <: Factorization{T}
    n::Cint
    nnzA::Cint
    descA::CuMatrixDescriptor
    info::SparseCholeskyInfo
    buffer::Union{CuPtr{Cvoid},CuVector{UInt8}}
end

function SparseCholesky(A::Union{CuSparseMatrixCSC{T,Cint},CuSparseMatrixCSR{T,Cint}}, index::Char='O') where T <: BlasFloat
    n = checksquare(A)
    nnzA = nnz(A)
    descA = CuMatrixDescriptor('G', 'L', 'N', index)
    info = SparseCholeskyInfo()
    buffer = CU_NULL
    F = SparseCholesky{T}(n, nnzA, descA, info, buffer)
    spcholesky_analyse(F, A)
    spcholesky_buffer(F, A)
    return F
end

# csrcholAnalysis
#
# cusolverStatus_t cusolverSpXcsrcholAnalysis(
#   cusolverSpHandle_t       handle,
#   int                      n,
#   int                      nnzA,
#   const cusparseMatDescr_t descrA,
#   const int *              csrRowPtrA,
#   const int *              csrColIndA,
#   csrcholInfo_t            info);
function spcholesky_analyse(F::SparseCholesky{T}, A::Union{CuSparseMatrixCSC{T,Cint},CuSparseMatrixCSR{T,Cint}}) where T <: BlasFloat
    if A isa CuSparseMatrixCSC
        cusolverSpXcsrcholAnalysis(sparse_handle(), F.n, F.nnzA, F.descA, A.colPtr, A.rowVal, F.info)
    else
        cusolverSpXcsrcholAnalysis(sparse_handle(), F.n, F.nnzA, F.descA, A.rowPtr, A.colVal, F.info)
    end
    return F
end

for (bname, fname, pname, elty, relty) in
    ((:cusolverSpScsrcholBufferInfo, :cusolverSpScsrcholFactor, :cusolverSpScsrcholZeroPivot, :Float32   , :Float32),
     (:cusolverSpDcsrcholBufferInfo, :cusolverSpDcsrcholFactor, :cusolverSpDcsrcholZeroPivot, :Float64   , :Float64),
     (:cusolverSpCcsrcholBufferInfo, :cusolverSpCcsrcholFactor, :cusolverSpCcsrcholZeroPivot, :ComplexF32, :Float32),
     (:cusolverSpZcsrcholBufferInfo, :cusolverSpZcsrcholFactor, :cusolverSpZcsrcholZeroPivot, :ComplexF64, :Float64))
    @eval begin
        # csrcholBufferInfo
        #
        # cusolverStatus_t cusolverSpScsrcholBufferInfo(
        #   cusolverSpHandle_t       handle,
        #   int                      n,
        #   int                      nnzA,
        #   const cusparseMatDescr_t descrA,
        #   const float *            csrValA,
        #   const int *              csrRowPtrA,
        #   const int *              csrColIndA,
        #   csrcholInfo_t            info,
        #   size_t *                 internalDataInBytes,
        #   size_t *                 workspaceInBytes);
        function spcholesky_buffer(F::SparseCholesky{$elty}, A::Union{CuSparseMatrixCSC{$elty,Cint},CuSparseMatrixCSR{$elty,Cint}})
            internalDataInBytes = Ref{Csize_t}(0)
            workspaceInBytes = Ref{Csize_t}(0)
            if A isa CuSparseMatrixCSC
                $bname(sparse_handle(), F.n, F.nnzA, F.descA, A.nzVal, A.colPtr, A.rowVal, F.info, internalDataInBytes, workspaceInBytes)
            else
                $bname(sparse_handle(), F.n, F.nnzA, F.descA, A.nzVal, A.rowPtr, A.colVal, F.info, internalDataInBytes, workspaceInBytes)
            end
            F.buffer = CuVector{UInt8}(undef, workspaceInBytes[])
            return F
        end

        # csrcholFactor
        #
        # cusolverStatus_t cusolverSpScsrcholFactor(
        #   cusolverSpHandle_t       handle,
        #   int                      n,
        #   int                      nnzA,
        #   const cusparseMatDescr_t descrA,
        #   const float *            csrValA,
        #   const int *              csrRowPtrA,
        #   const int *              csrColIndA,
        #   csrcholInfo_t            info,
        #   void *                   pBuffer);
        #
        # csrcholZeroPivot
        #
        # cusolverStatus_t cusolverSpScsrcholZeroPivot(
        #   cusolverSpHandle_t handle,
        #   csrcholInfo_t      info,
        #   float              tol,
        #   int *              position);
        function spcholesky_factorise(F::SparseCholesky{$elty}, A::Union{CuSparseMatrixCSC{$elty,Cint},CuSparseMatrixCSR{$elty,Cint}}, tol::$relty)
            if A isa CuSparseMatrixCSC
                nzval = $elty <: Complex ? conj(A.nzVal) : A.nzVal
                $fname(sparse_handle(), F.n, F.nnzA, F.descA, nzval, A.colPtr, A.rowVal, F.info, F.buffer)
            else
                $fname(sparse_handle(), F.n, F.nnzA, F.descA, A.nzVal, A.rowPtr, A.colVal, F.info, F.buffer)
            end
            singularity = Ref{Cint}(0)
            $pname(sparse_handle(), F.info, tol, singularity)
            (singularity[] ≥ 0) && throw(SingularException(singularity[]))
            return F
        end
    end
end

for (sname, dname, elty, relty) in ((:cusolverSpScsrcholSolve, :cusolverSpScsrcholDiag, :Float32   , :Float32),
                                    (:cusolverSpDcsrcholSolve, :cusolverSpDcsrcholDiag, :Float64   , :Float64),
                                    (:cusolverSpCcsrcholSolve, :cusolverSpCcsrcholDiag, :ComplexF32, :Float32),
                                    (:cusolverSpZcsrcholSolve, :cusolverSpZcsrcholDiag, :ComplexF64, :Float64))
    @eval begin
        # csrcholSolve
        #
        # cusolverStatus_t cusolverSpZcsrcholSolve(
        #   cusolverSpHandle_t     handle,
        #   int                    n,
        #   const cuDoubleComplex *b,
        #   cuDoubleComplex *      x,
        #   csrcholInfo_t          info,
        #   void *                 pBuffer);
        function spcholesky_solve(F::SparseCholesky{$elty}, b::CuVector{$elty}, x::CuVector{$elty})        
            $sname(sparse_handle(), F.n, b, x, F.info, F.buffer)
            return x
        end

        function spcholesky_solve(F::SparseCholesky{$elty}, B::CuMatrix{$elty}, X::CuMatrix{$elty})
            n, p = size(B)
            for j=1:p
                $sname(sparse_handle(), F.n, view(B,:,j), view(X,:,j), F.info, F.buffer)
            end
            return X
        end

        # csrcholDiag
        #
        # cusolverStatus_t cusolverSpCcsrcholDiag(
        #   cusolverSpHandle_t handle,
        #   csrcholInfo_t      info,
        #   float *            diag);
        function spcholesky_diag(F::SparseCholesky{$elty}, diag::CuVector{$relty})
            $dname(sparse_handle(), F.info, diag)
            return diag
        end
    end
end
