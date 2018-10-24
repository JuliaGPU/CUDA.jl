#enum cusparseStatus_t
#error messages from CUSPARSE

"""
Status messages from CUSPARSE's C API.
"""
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
"""
Perform operation on indices only (`CUSPARSE_ACTION_SYMBOLIC`) or
on both data and indices (`CUSPARSE_ACTION_NUMERIC`). Used in
conversion routines.
"""
const cusparseAction_t = UInt32
const CUSPARSE_ACTION_SYMBOLIC = 0
const CUSPARSE_ACTION_NUMERIC  = 1

#enum cusparseDirection_t
"""
Parse dense matrix by rows (`CUSPARSE_DIRECTION_ROW`) or columns
(`CUSPARSE_DIRECTION_COL`) to compute its number of non-zeros.
"""
const cusparseDirection_t = UInt32
const CUSPARSE_DIRECTION_ROW = 0
const CUSPARSE_DIRECTION_COL = 1

#enum cusparseHybPartition_t
"""
How to partition the HYB matrix in a [`CudaSparseMatrixHYB`](@ref).
There are three choices:
* `CUSPARSE_HYB_PARTITION_AUTO` - let CUSPARSE decide internally for best performance.
* `CUSPARSE_HYB_PARTITION_USER` - set the partition manually in the conversion function.
* `CUSPARSE_HYB_PARTITION_MAX` - use the maximum partition, putting the matrix in ELL format.
"""
const cusparseHybPartition_t = UInt32
const CUSPARSE_HYB_PARTITION_AUTO = 0
const CUSPARSE_HYB_PARTITION_USER = 1
const CUSPARSE_HYB_PARTITION_MAX  = 2

#enum cusparseFillMode_t
"""
Determines if a symmetric/Hermitian/triangular matrix has its upper
(`CUSPARSE_FILL_MODE_UPPER`) or lower (`CUSPARSE_FILL_MODE_LOWER`)
triangle filled.
"""
const cusparseFillMode_t = UInt32
const CUSPARSE_FILL_MODE_LOWER = 0
const CUSPARSE_FILL_MODE_UPPER = 1

#enum cusparseDiagType_t
"""
Determines if the diagonal of a matrix is all ones (`CUSPARSE_DIAG_TYPE_UNIT`)
or not all ones (`CUSPARSE_DIAG_TYPE_NON_UNIT`).
"""
const cusparseDiagType_t = UInt32
const CUSPARSE_DIAG_TYPE_NON_UNIT = 0
const CUSPARSE_DIAG_TYPE_UNIT     = 1

#enum cusparsePointerMode_t
"""
Determines if scalar arguments to a function are present on the host CPU
(`CUSPARSE_POINTER_MODE_HOST`) or on the GPU (`CUSPARSE_POINTER_MODE_DEVICE`).
"""
const cusparsePointerMode_t = UInt32
const CUSPARSE_POINTER_MODE_HOST   = 0
const CUSPARSE_POINTER_MODE_DEVICE = 1

#enum cusparseOperation_t
"""
Determines whether to perform an operation, such as a matrix multiplication
or solve, on the matrix as-is (`CUSPARSE_OPERATION_NON_TRANSPOSE`), on the
matrix's transpose (`CUSPARSE_OPERATION_TRANSPOSE`), or on its conjugate
transpose (`CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE`).
"""
const cusparseOperation_t = UInt32
const CUSPARSE_OPERATION_NON_TRANSPOSE       = 0
const CUSPARSE_OPERATION_TRANSPOSE           = 1
const CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2

#enum cusparseMatrixType_t
"""
Indicates whether a matrix is a general matrix (`CUSPARSE_MATRIX_TYPE_GENERAL`),
symmetric (`CUSPARSE_MATRIX_TYPE_SYMMETRIC`), Hermitian
(`CUSPARSE_MATRIX_TYPE_HERMITIAN`), or triangular
(`CUSPARSE_MATRIX_TYPE_TRIANGULAR`). Note that for some matrix types
(those in [`CompressedSparse`](@ref)), this can be inferred for some function
calls.
"""
const cusparseMatrixType_t = UInt32
const CUSPARSE_MATRIX_TYPE_GENERAL    = 0
const CUSPARSE_MATRIX_TYPE_SYMMETRIC  = 1
const CUSPARSE_MATRIX_TYPE_HERMITIAN  = 2
const CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3

#enum cusparseSolvePolicy_t
"""
Indicates whether to keep level info in solvers (`CUSPARSE_SOLVE_POLICY_USE_LEVEL`)
or whether to not use it (`CUSPARSE_SOLVE_POLICY_NO_LEVEL`).
"""
const cusparseSolvePolicy_t = UInt32
const CUSPARSE_SOLVE_POLICY_NO_LEVEL  = 0
const CUSPARSE_SOLVE_POLICY_USE_LEVEL = 1

#enum cusparseIndexBase_t
"""
Indicates whether a sparse object is zero-indexed (`CUSPARSE_INDEX_BASE_ZERO`)
or one-indexed (`CUSPARSE_INDEX_BASE_ONE`). CUSPARSE.jl supports both. Julia
sparse matrices are one-indexed, but you may wish to pass matrices from other
libraries which use zero-indexing (e.g. C language ODE solvers).
"""
const cusparseIndexBase_t = UInt32
const CUSPARSE_INDEX_BASE_ZERO = 0
const CUSPARSE_INDEX_BASE_ONE  = 1

#struct cusparseMatDescr_t
"""
Describes shape and properties of a CUSPARSE matrix. A convenience wrapper.

Contains:
* `MatrixType` - a [`cusparseMatrixType_t`](@ref)
* `FillMode` - a [`cusparseFillMode_t`](@ref)
* `DiagType` - a [`cusparseDiagType_t`](@ref)
* `IndexBase` - a [`cusparseIndexBase_t`](@ref)
"""
struct cusparseMatDescr_t
    MatrixType::cusparseMatrixType_t
    FillMode::cusparseFillMode_t
    DiagType::cusparseDiagType_t
    IndexBase::cusparseIndexBase_t
    function cusparseMatDescr_t(MatrixType,FillMode,DiagType,IndexBase)
        new(MatrixType,FillMode,DiagType,IndexBase)
    end
end

"""
An opaque struct containing information about the solution approach
CUSPARSE will take. Generated by [`sv_analysis`](@ref) or
[`sm_analysis`](@ref) and passed to [`sv_solve!`](@ref), [`sm_solve`](@ref),
[`ic0!`](@ref), or [`ilu0!`](@ref).
"""
const cusparseSolveAnalysisInfo_t = Ptr{Cvoid}
const bsrsm2Info_t = Ptr{Cvoid}
const bsrsv2Info_t = Ptr{Cvoid}
const csrsv2Info_t = Ptr{Cvoid}
const csric02Info_t = Ptr{Cvoid}
const csrilu02Info_t = Ptr{Cvoid}
const bsric02Info_t = Ptr{Cvoid}
const bsrilu02Info_t = Ptr{Cvoid}

const cusparseContext = Cvoid
const cusparseHandle_t = Ptr{cusparseContext}

#complex numbers

const cuComplex = Complex{Float32}
const cuDoubleComplex = Complex{Float64}

const CusparseFloat = Union{Float64,Float32,ComplexF64,ComplexF32}
const CusparseReal = Union{Float64,Float32}
const CusparseComplex = Union{ComplexF64,ComplexF32}
