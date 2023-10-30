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

mutable struct SparseQR{T <: BlasFloat}
  n::Cint
  m::Cint
  nnzA::Cint
  mu::T
  handle::cusolverSpHandle_t
  descA::CuMatrixDescriptor
  info::SparseQRInfo
  buffer::Union{CuPtr{Cvoid},CuVector{UInt8}}
end

function SparseQR(A::CuSparseMatrixCSR{T,Cint}, index::Char='O') where T <: BlasFloat
    m,n = size(A)
    nnzA = nnz(A)
    mu = zero(T)
    handle = sparse_handle()
    descA = CuMatrixDescriptor('G', 'L', 'N', index)
    handle = sparse_handle()
    info = SparseQRInfo()
    buffer = CU_NULL
    F = SparseQR{T}(n, m, nnzA, mu, handle, descA, info, buffer)
    spqr_analyse(F, A)
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
    cusolverSpXcsrqrAnalysis(F.handle, F.m, F.n, F.nnzA, F.descA, A.rowPtr, A.colVal, F.info)
    return F
end

#csrqrSetup
for (fname, elty) in ((:cusolverSpScsrqrSetup, :Float32),
                      (:cusolverSpDcsrqrSetup, :Float64),
                      (:cusolverSpCcsrqrSetup, :ComplexF32),
                      (:cusolverSpZcsrqrSetup, :ComplexF64))
    @eval begin
        # cusolverStatus_t cusolverSpScsrqrSetup(
        # cusolverSpHandle_t       handle,
        # int                      m,
        # int                      n,
        # int                      nnzA,
        # const cusparseMatDescr_t descrA,
        # const float *            csrValA,
        # const int *              csrRowPtrA,
        # const int *              csrColIndA,
        # float                    mu,
        # csrqrInfo_t              info);
        function spqr_setup(F::SparseQR{$elty}, A::CuSparseMatrixCSR{$elty,Cint})
            $fname(F.handle, F.m, F.n, F.nnzA, F.descA, A.nzVal, A.rowPtr, A.colVal, F.mu, F.info)
            return F
        end
    end
end

for (bname, fname, sname, pname, elty, relty) in
    ((:cusolverSpScsrqrBufferInfo, :cusolverSpScsrqrFactor, :cusolverSpScsrqrSolve, :cusolverSpScsrqrZeroPivot, :Float32   , :Float32),
     (:cusolverSpDcsrqrBufferInfo, :cusolverSpDcsrqrFactor, :cusolverSpDcsrqrSolve, :cusolverSpDcsrqrZeroPivot, :Float64   , :Float64),
     (:cusolverSpCcsrqrBufferInfo, :cusolverSpCcsrqrFactor, :cusolverSpCcsrqrSolve, :cusolverSpCcsrqrZeroPivot, :ComplexF32, :Float32),
     (:cusolverSpZcsrqrBufferInfo, :cusolverSpZcsrqrFactor, :cusolverSpZcsrqrSolve, :cusolverSpZcsrqrZeroPivot, :ComplexF64, :Float64))
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
            $bname(F.handle, F.m, F.n, F.nnzA, F.descA, A.nzVal, A.rowPtr, A.colVal, F.info, internalDataInBytes, workspaceInBytes)
            # TODO: allocate buffer?
            F.buffer = CuVector{UInt8}(undef, workspaceInBytes[])
            return F
        end

        # csrqrFactor
        #
        # cusolverStatus_t cusolverSpScsrqrFactor(
        # cusolverSpHandle_t handle,
        # int                m,
        # int                n,
        # int                nnzA,
        # float *            b,
        # float *            x,
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
        function spqr_factorise(F::SparseQR{$elty}, tol::$relty)
            $fname(F.handle, F.m, F.n, F.nnzA, CU_NULL, CU_NULL, F.info, F.buffer)
            singularity = Ref{Cint}(0)
            $pname(F.handle, F.info, tol, singularity)
            (singularity[] ≥ 0) && throw(SingularException(singularity[]))
            return F
        end

        function spqr_factorise_solve(F::SparseQR{$elty}, b::CuVecOrMat{$elty}, x::CuVecOrMat{$elty}, tol::$relty)
            $fname(F.handle, F.m, F.n, F.nnzA, b, x, F.info, F.buffer)
            singularity = Ref{Cint}(0)
            $pname(F.handle, F.info, tol, singularity)
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
        function spqr_solve(F::SparseQR{$elty}, b::CuVecOrMat{$elty}, x::CuVecOrMat{$elty})
            $sname(F.handle, F.m, F.n, b, x, F.info, F.buffer)
            return x
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

mutable struct SparseCholesky{T <: BlasFloat}
    n::Cint
    nnzA::Cint
    handle::cusolverSpHandle_t
    descA::CuMatrixDescriptor
    info::SparseCholeskyInfo
    buffer::Union{CuPtr{Cvoid},CuVector{UInt8}}
end

function SparseCholesky(A::CuSparseMatrixCSR{T,Cint}, index::Char='O') where T <: BlasFloat
    n = checksquare(A)
    nnzA = nnz(A)
    handle = sparse_handle()
    descA = CuMatrixDescriptor('G', 'L', 'N', index)
    info = SparseCholeskyInfo()
    buffer = CU_NULL
    F = SparseCholesky{T}(n, nnzA, handle, descA, info, buffer)
    spcholesky_analyse(F, A)
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
function spcholesky_analyse(F::SparseCholesky{T}, A::CuSparseMatrixCSR{T}) where T <: BlasFloat
    cusolverSpXcsrcholAnalysis(F.handle, F.n, F.nnzA, F.descA, A.rowPtr, A.colVal, F.info)
    return F
end

for (bname, fname, pname, sname, dname, elty, relty) in
    ((:cusolverSpScsrcholBufferInfo, :cusolverSpScsrcholFactor, :cusolverSpScsrcholZeroPivot, :cusolverSpScsrcholSolve, :cusolverSpScsrcholDiag, :Float32   , :Float32),
     (:cusolverSpDcsrcholBufferInfo, :cusolverSpDcsrcholFactor, :cusolverSpDcsrcholZeroPivot, :cusolverSpDcsrcholSolve, :cusolverSpDcsrcholDiag, :Float64   , :Float64),
     (:cusolverSpCcsrcholBufferInfo, :cusolverSpCcsrcholFactor, :cusolverSpCcsrcholZeroPivot, :cusolverSpCcsrcholSolve, :cusolverSpCcsrcholDiag, :ComplexF32, :Float32),
     (:cusolverSpZcsrcholBufferInfo, :cusolverSpZcsrcholFactor, :cusolverSpZcsrcholZeroPivot, :cusolverSpZcsrcholSolve, :cusolverSpZcsrcholDiag, :ComplexF64, :Float64))
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
        function spcholesky_buffer(F::SparseCholesky{$elty}, A::CuSparseMatrixCSR{$elty})
            internalDataInBytes = Ref{Csize_t}(0)
            workspaceInBytes = Ref{Csize_t}(0)
            $bname(F.handle, F.n, F.nnzA, F.descA, A.nzVal, A.rowPtr, A.colVal, F.info, internalDataInBytes, workspaceInBytes)
            # TODO: allocate buffer?
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
        function spcholesky_factorise(F::SparseCholesky{$elty}, A::CuSparseMatrixCSR{$elty}, tol::$relty)
            $fname(F.handle, F.n, F.nnzA, F.descA, A.nzVal, A.rowPtr, A.colVal, F.info, F.buffer)
            singularity = Ref{Cint}(0)
            $pname(F.handle, F.info, tol, singularity)
            (singularity[] ≥ 0) && throw(SingularException(singularity[]))
            return F
        end

        # csrcholSolve
        #
        # cusolverStatus_t cusolverSpZcsrcholSolve(
        #   cusolverSpHandle_t     handle,
        #   int                    n,
        #   const cuDoubleComplex *b,
        #   cuDoubleComplex *      x,
        #   csrcholInfo_t          info,
        #   void *                 pBuffer);
        function spcholesky_solve(F::SparseCholesky{$elty}, b::CuVecOrMat{$elty}, x::CuVecOrMat{$elty})
            $sname(F.handle, F.n, b, x, F.info, F.buffer)
            return x
        end

        # csrcholDiag
        #
        # cusolverStatus_t cusolverSpCcsrcholDiag(
        #   cusolverSpHandle_t handle,
        #   csrcholInfo_t      info,
        #   float *            diag);
        function spcholesky_diag(F::SparseCholesky{$elty}, diag::CuVector{$relty})
            $dname(F.handle, F.info, diag)
            return diag
        end
    end
end
