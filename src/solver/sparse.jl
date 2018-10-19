const cusparseStatus_t = UInt32
const CUSPARSE_STATUS_SUCCESS                   = 0
const CUSPARSE_STATUS_NOT_INITIALIZED           = 1
const CUSPARSE_STATUS_ALLOC_FAILED              = 2
const CUSPARSE_STATUS_INVALID_VALUE             = 3
const CUSPARSE_STATUS_ARCH_MISMATCH             = 4
const CUSPARSE_STATUS_MAPPING_ERROR             = 5
const CUSPARSE_STATUS_EXECUTION_FAILED          = 6
const CUSPARSE_STATUS_INTERNAL_ERROR            = 7
const CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8

#enum cusparseAction_t
#"""
#Perform operation on indices only (`CUSPARSE_ACTION_SYMBOLIC`) or
#on both data and indices (`CUSPARSE_ACTION_NUMERIC`). Used in
#conversion routines.
#"""
const cusparseAction_t = UInt32
const CUSPARSE_ACTION_SYMBOLIC = 0
const CUSPARSE_ACTION_NUMERIC  = 1
#
##enum cusparseDirection_t
#"""
#Parse dense matrix by rows (`CUSPARSE_DIRECTION_ROW`) or columns
#(`CUSPARSE_DIRECTION_COL`) to compute its number of non-zeros.
#"""
const cusparseDirection_t = UInt32
const CUSPARSE_DIRECTION_ROW = 0
const CUSPARSE_DIRECTION_COL = 1
#
##enum cusparseHybPartition_t
#"""
#How to partition the HYB matrix in a [`CudaSparseMatrixHYB`](@ref).
#There are three choices:
#* `CUSPARSE_HYB_PARTITION_AUTO` - let CUSPARSE decide internally for best performance.
#* `CUSPARSE_HYB_PARTITION_USER` - set the partition manually in the conversion function.
#* `CUSPARSE_HYB_PARTITION_MAX` - use the maximum partition, putting the matrix in ELL format.
#"""
const cusparseHybPartition_t = UInt32
const CUSPARSE_HYB_PARTITION_AUTO = 0
const CUSPARSE_HYB_PARTITION_USER = 1
const CUSPARSE_HYB_PARTITION_MAX  = 2
#
##enum cusparseFillMode_t
#"""
#Determines if a symmetric/Hermitian/triangular matrix has its upper
#(`CUSPARSE_FILL_MODE_UPPER`) or lower (`CUSPARSE_FILL_MODE_LOWER`)
#triangle filled.
#"""
const cusparseFillMode_t = UInt32
const CUSPARSE_FILL_MODE_LOWER = 0
const CUSPARSE_FILL_MODE_UPPER = 1
#
##enum cusparseDiagType_t
#"""
#Determines if the diagonal of a matrix is all ones (`CUSPARSE_DIAG_TYPE_UNIT`)
#or not all ones (`CUSPARSE_DIAG_TYPE_NON_UNIT`).
#"""
const cusparseDiagType_t = UInt32
const CUSPARSE_DIAG_TYPE_NON_UNIT = 0
const CUSPARSE_DIAG_TYPE_UNIT     = 1
#
##enum cusparsePointerMode_t
#"""
#Determines if scalar arguments to a function are present on the host CPU
#(`CUSPARSE_POINTER_MODE_HOST`) or on the GPU (`CUSPARSE_POINTER_MODE_DEVICE`).
#"""
const cusparsePointerMode_t = UInt32
const CUSPARSE_POINTER_MODE_HOST   = 0
const CUSPARSE_POINTER_MODE_DEVICE = 1
#
##enum cusparseOperation_t
#"""
#Determines whether to perform an operation, such as a matrix multiplication
#or solve, on the matrix as-is (`CUSPARSE_OPERATION_NON_TRANSPOSE`), on the
#matrix's transpose (`CUSPARSE_OPERATION_TRANSPOSE`), or on its conjugate
#transpose (`CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE`).
#"""
const cusparseOperation_t = UInt32
const CUSPARSE_OPERATION_NON_TRANSPOSE       = 0
const CUSPARSE_OPERATION_TRANSPOSE           = 1
const CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
#
##enum cusparseMatrixType_t
#"""
#Indicates whether a matrix is a general matrix (`CUSPARSE_MATRIX_TYPE_GENERAL`),
#symmetric (`CUSPARSE_MATRIX_TYPE_SYMMETRIC`), Hermitian
#(`CUSPARSE_MATRIX_TYPE_HERMITIAN`), or triangular
#(`CUSPARSE_MATRIX_TYPE_TRIANGULAR`). Note that for some matrix types
#(those in [`CompressedSparse`](@ref)), this can be inferred for some function
#calls.
#"""
const cusparseMatrixType_t = UInt32
const CUSPARSE_MATRIX_TYPE_GENERAL    = 0
const CUSPARSE_MATRIX_TYPE_SYMMETRIC  = 1
const CUSPARSE_MATRIX_TYPE_HERMITIAN  = 2
const CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3
#
##enum cusparseSolvePolicy_t
#"""
#Indicates whether to keep level info in solvers (`CUSPARSE_SOLVE_POLICY_USE_LEVEL`)
#or whether to not use it (`CUSPARSE_SOLVE_POLICY_NO_LEVEL`).
#"""
const cusparseSolvePolicy_t = UInt32
const CUSPARSE_SOLVE_POLICY_NO_LEVEL  = 0
const CUSPARSE_SOLVE_POLICY_USE_LEVEL = 1
#
##enum cusparseIndexBase_t
#"""
#Indicates whether a sparse object is zero-indexed (`CUSPARSE_INDEX_BASE_ZERO`)
#or one-indexed (`CUSPARSE_INDEX_BASE_ONE`). CUSPARSE.jl supports both. Julia
#sparse matrices are one-indexed, but you may wish to pass matrices from other
#libraries which use zero-indexing (e.g. C language ODE solvers).
#"""
const cusparseIndexBase_t = UInt32
const CUSPARSE_INDEX_BASE_ZERO = 0
const CUSPARSE_INDEX_BASE_ONE  = 1

struct cusparseMatDescr_t
    MatrixType::cusparseMatrixType_t
    FillMode::cusparseFillMode_t
    DiagType::cusparseDiagType_t
    IndexBase::cusparseIndexBase_t
    function cusparseMatDescr_t(MatrixType,FillMode,DiagType,IndexBase)
        new(MatrixType,FillMode,DiagType,IndexBase)
    end
end


function cusparseop(trans::Char)
    if trans == 'N'
        return CUSPARSE_OPERATION_NON_TRANSPOSE
    end
    if trans == 'T'
        return CUSPARSE_OPERATION_TRANSPOSE
    end
    if trans == 'C'
        return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
    end
    throw(ArgumentError("unknown cusparse operation $trans"))
end

function cusparsetype(mattype::Char)
    if mattype == 'G'
        return CUSPARSE_MATRIX_TYPE_GENERAL
    end
    if mattype == 'T'
        return CUSPARSE_MATRIX_TYPE_TRIANGULAR
    end
    if mattype == 'S'
        return CUSPARSE_MATRIX_TYPE_SYMMETRIC
    end
    if mattype == 'H'
        return CUSPARSE_MATRIX_TYPE_HERMITIAN
    end
    throw(ArgumentError("unknown cusparse matrix type $mattype"))
end

function cusparsefill(uplo::Char)
    if uplo == 'U'
        return CUSPARSE_FILL_MODE_UPPER
    end
    if uplo == 'L'
        return CUSPARSE_FILL_MODE_LOWER
    end
    throw(ArgumentError("unknown cusparse fill mode $uplo"))
end

function cusparsediag(diag::Char)
    if diag == 'U'
        return CUSPARSE_DIAG_TYPE_UNIT
    end
    if diag == 'N'
        return CUSPARSE_DIAG_TYPE_NON_UNIT
    end
    throw(ArgumentError("unknown cusparse diag mode $diag"))
end

function cusparseindex(index::Char)
    if index == 'Z'
        return CUSPARSE_INDEX_BASE_ZERO
    end
    if index == 'O'
        return CUSPARSE_INDEX_BASE_ONE
    end
    throw(ArgumentError("unknown cusparse index base"))
end

function cusparsedir(dir::Char)
    if dir == 'R'
        return CUSPARSE_DIRECTION_ROW
    end
    if dir == 'C'
        return CUSPARSE_DIRECTION_COL
    end
    throw(ArgumentError("unknown cusparse direction $dir"))
end

#csrlsvlu 
for (fname, elty, relty) in ((:cusolverSpScsrlsvluHost, :Float32, :Float32),
                             (:cusolverSpDcsrlsvluHost, :Float64, :Float64),
                             (:cusolverSpCcsrlsvluHost, :ComplexF32, :Float32),
                             (:cusolverSpZcsrlsvluHost, :ComplexF64, Float64))
    @eval begin
        function csrlsvlu!(A::SparseMatrixCSC{$elty},
                           b::Vector{$elty},
                           x::Vector{$elty},
                           tol::$relty,
                           reorder::Cint,
                           inda::Char)
            cuinda = cusparseindex(inda)
            n = size(A,1)
            if size(A,2) != n
                throw(DimensionMismatch("LU factorization is only possible for square matrices!"))
            end
            if size(A,2) != length(b)
                throw(DimensionMismatch("second dimension of A, $(size(A,2)), must match the length of b, $(length(b))"))
            end
            if length(x) != length(b)
                throw(DimensionMismatch("length of x, $(length(x)), must match the length of b, $(length(b))"))
            end
            Mat = transpose(A)
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            rcudesca = Ref{cusparseMatDescr_t}(cudesca)
            singularity = zeros(Cint,1)
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverSpHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, Ptr{$elty}, $relty, Cint, Ptr{$elty},
                               Ptr{Cint}),
                              cusolverSphandle[1], n, length(A.nzval), rcudesca,
                              Mat.nzval, convert(Vector{Cint},Mat.colptr),
                              convert(Vector{Cint},Mat.rowval), b, tol, reorder,
                              x, singularity))
            if singularity[1] != -1
                throw(Base.LinAlg.SingularException(singularity[1]))
            end
            x
        end
    end
end

#csrlsvqr 
for (fname, elty, relty) in ((:cusolverSpScsrlsvqr, :Float32, :Float32),
                             (:cusolverSpDcsrlsvqr, :Float64, :Float64),
                             (:cusolverSpCcsrlsvqr, :ComplexF32, :Float32),
                             (:cusolverSpZcsrlsvqr, :ComplexF64, Float64))
    @eval begin
        function csrlsvqr!(A::CudaSparseMatrixCSR{$elty},
                           b::CudaVector{$elty},
                           x::CudaVector{$elty},
                           tol::$relty,
                           reorder::Cint,
                           inda::Char)
            cuinda = cusparseindex(inda)
            n = size(A,1)
            if size(A,2) != n
                throw(DimensionMismatch("QR factorization is only possible for square matrices!"))
            end
            if size(A,2) != length(b)
                throw(DimensionMismatch("second dimension of A, $(size(A,2)), must match the length of b, $(length(b))"))
            end
            if length(x) != length(b)
                throw(DimensionMismatch("length of x, $(length(x)), must match the length of b, $(length(b))"))
            end
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            rcudesca = Ref{cusparseMatDescr_t}(cudesca)
            singularity = Array{Cint}(1)
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverSpHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, Ptr{$elty}, $relty, Cint, Ptr{$elty},
                               Ptr{Cint}),
                              cusolverSphandle[1], n, A.nnz, rcudesca, A.nzVal,
                              A.rowPtr, A.colVal, b, tol, reorder, x, singularity))
            if singularity[1] != -1
                throw(Base.LinAlg.SingularException(singularity[1]))
            end
            x
        end
    end
end

#csrlsvchol
for (fname, elty, relty) in ((:cusolverSpScsrlsvchol, :Float32, :Float32),
                             (:cusolverSpDcsrlsvchol, :Float64, :Float64),
                             (:cusolverSpCcsrlsvchol, :ComplexF32, :Float32),
                             (:cusolverSpZcsrlsvchol, :ComplexF64, Float64))
    @eval begin
        function csrlsvchol!(A::CudaSparseMatrixCSR{$elty},
                             b::CudaVector{$elty},
                             x::CudaVector{$elty},
                             tol::$relty,
                             reorder::Cint,
                             inda::Char)
            cuinda = cusparseindex(inda)
            n      = size(A,1)
            if size(A,2) != n
                throw(DimensionMismatch("Cholesky factorization is only possible for square matrices!"))
            end
            if size(A,2) != length(b)
                throw(DimensionMismatch("second dimension of A, $(size(A,2)), must match the length of b, $(length(b))"))
            end
            if length(x) != length(b)
                throw(DimensionMismatch("length of x, $(length(x)), must match the length of b, $(length(b))"))
            end

            cudesca     = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            rcudesca = Ref{cusparseMatDescr_t}(cudesca)
            singularity = zeros(Cint,1)
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverSpHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, Ptr{$elty}, $relty, Cint, Ptr{$elty},
                               Ptr{Cint}),
                              cusolverSphandle[1], n, A.nnz, rcudesca, A.nzVal,
                              A.rowPtr, A.colVal, b, tol, reorder, x, singularity))
            if singularity[1] != -1
                throw(Base.LinAlg.SingularException(singularity[1]))
            end
            x
        end
    end
end

#csrlsqvqr 
for (fname, elty, relty) in ((:cusolverSpScsrlsqvqrHost, :Float32, :Float32),
                             (:cusolverSpDcsrlsqvqrHost, :Float64, :Float64),
                             (:cusolverSpCcsrlsqvqrHost, :ComplexF32, :Float32),
                             (:cusolverSpZcsrlsqvqrHost, :ComplexF64, Float64))
    @eval begin
        function csrlsqvqr!(A::SparseMatrixCSC{$elty},
                            b::Vector{$elty},
                            x::Vector{$elty},
                            tol::$relty,
                            inda::Char)
            cuinda = cusparseindex(inda)
            m,n    = size(A)
            if m < n
                throw(DimensionMismatch("csrlsqvqr only works when the first dimension of A, $m, is greater than or equal to the second dimension of A, $n"))
            end
            if size(A,2) != length(b)
                throw(DimensionMismatch("second dimension of A, $(size(A,2)), must match the length of b, $(length(b))"))
            end
            if length(x) != length(b)
                throw(DimensionMismatch("length of x, $(length(x)), must match the length of b, $(length(b))"))
            end
            cudesca  = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            rcudesca = Ref{cusparseMatDescr_t}(cudesca)
            p        = zeros(Cint,n)
            min_norm = zeros($relty,1)
            rankA    = zeros(Cint,1)
            Mat      = transpose(A)
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverSpHandle_t, Cint, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, Ptr{$elty}, $relty, Ptr{Cint},
                               Ptr{$elty}, Ptr{Cint}, Ptr{$relty}),
                              cusolverSphandle[1], m, n, length(A.nzval),
                              rcudesca, Mat.nzval, convert(Vector{Cint},Mat.colptr),
                              convert(Vector{Cint},Mat.rowval), b,
                              tol, rankA, x, p, min_norm))
            x, rankA[1], p, min_norm[1]
        end
    end
end

#csreigvsi 
for (fname, elty, relty) in ((:cusolverSpScsreigvsi, :Float32, :Float32),
                             (:cusolverSpDcsreigvsi, :Float64, :Float64),
                             (:cusolverSpCcsreigvsi, :ComplexF32, :Float32),
                             (:cusolverSpZcsreigvsi, :ComplexF64, Float64))
    @eval begin
        function csreigvsi(A::CudaSparseMatrixCSR{$elty},
                           μ_0::$elty,
                           x_0::CudaVector{$elty},
                           tol::$relty,
                           maxite::Cint,
                           inda::Char)
            cuinda = cusparseindex(inda)
            m,n    = size(A)
            if m != n
                throw(DimensionMismatch("A must be square!"))
            end
            if n != length(x_0)
                throw(DimensionMismatch("second dimension of A, $(size(A,2)), must match the length of x_0, $(length(x_0))"))
            end
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            rcudesca = Ref{cusparseMatDescr_t}(cudesca)
            x       = copy(x_0) 
            μ       = CudaArray(zeros($elty,1)) 
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverSpHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$elty}, Ptr{Cint},
                               Ptr{Cint}, $elty, Ptr{$elty}, Cint,
                               $relty, Ptr{$elty}, Ptr{$elty}),
                              cusolverSphandle[1], n, A.nnz, rcudesca, A.nzVal,
                              A.rowPtr, A.colVal, μ_0, x_0, maxite, tol, μ, x))
            to_host(μ)[1], x
        end
    end
end

#csreigs
for (fname, elty, relty) in ((:cusolverSpScsreigsHost, :ComplexF32, :Float32),
                             (:cusolverSpDcsreigsHost, :ComplexF64, :Float64),
                             (:cusolverSpCcsreigsHost, :ComplexF32, :ComplexF32),
                             (:cusolverSpZcsreigsHost, :ComplexF64, :ComplexF64))
    @eval begin
        function csreigs(A::SparseMatrixCSC{$relty},
                         lbc::$elty,
                         ruc::$elty,
                         inda::Char)
            cuinda = cusparseindex(inda)
            m,n    = size(A)
            if m != n
                throw(DimensionMismatch("A must be square!"))
            end
            cudesca = cusparseMatDescr_t(CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, cuinda)
            rcudesca = Ref{cusparseMatDescr_t}(cudesca)
            numeigs = zeros(Cint,1)
            Mat     = A.'
            statuscheck(ccall(($(string(fname)),libcusolver), cusolverStatus_t,
                              (cusolverSpHandle_t, Cint, Cint,
                               Ptr{cusparseMatDescr_t}, Ptr{$relty}, Ptr{Cint},
                               Ptr{Cint}, $elty, $elty, Ptr{Cint}),
                              cusolverSphandle[1], n, length(A.nzval), rcudesca,
                              Mat.nzval, convert(Vector{Cint},Mat.colptr),
                              convert(Vector{Cint},Mat.rowval), lbc, ruc, numeigs))
            numeigs[1]
        end
    end
end
