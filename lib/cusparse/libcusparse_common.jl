# Automatically generated using Clang.jl

const CUSPARSE_VER_MAJOR = 11
const CUSPARSE_VER_MINOR = 4
const CUSPARSE_VER_PATCH = 1
const CUSPARSE_VER_BUILD = 1152
const CUSPARSE_VERSION = CUSPARSE_VER_MAJOR * 1000 + CUSPARSE_VER_MINOR * 100 + CUSPARSE_VER_PATCH

# Skipping MacroDefinition: CUSPARSE_DEPRECATED ( new_func ) __attribute__ ( ( deprecated ( "please use " # new_func " instead" ) ) )
# Skipping MacroDefinition: CUSPARSE_DEPRECATED_ENUM ( new_enum ) __attribute__ ( ( deprecated ( "please use " # new_enum " instead" ) ) )

const cusparseContext = Cvoid
const cusparseHandle_t = Ptr{cusparseContext}
const cusparseMatDescr = Cvoid
const cusparseMatDescr_t = Ptr{cusparseMatDescr}
const csrsv2Info = Cvoid
const csrsv2Info_t = Ptr{csrsv2Info}
const csrsm2Info = Cvoid
const csrsm2Info_t = Ptr{csrsm2Info}
const bsrsv2Info = Cvoid
const bsrsv2Info_t = Ptr{bsrsv2Info}
const bsrsm2Info = Cvoid
const bsrsm2Info_t = Ptr{bsrsm2Info}
const csric02Info = Cvoid
const csric02Info_t = Ptr{csric02Info}
const bsric02Info = Cvoid
const bsric02Info_t = Ptr{bsric02Info}
const csrilu02Info = Cvoid
const csrilu02Info_t = Ptr{csrilu02Info}
const bsrilu02Info = Cvoid
const bsrilu02Info_t = Ptr{bsrilu02Info}
const csrgemm2Info = Cvoid
const csrgemm2Info_t = Ptr{csrgemm2Info}
const csru2csrInfo = Cvoid
const csru2csrInfo_t = Ptr{csru2csrInfo}
const cusparseColorInfo = Cvoid
const cusparseColorInfo_t = Ptr{cusparseColorInfo}
const pruneInfo = Cvoid
const pruneInfo_t = Ptr{pruneInfo}

@cenum cusparseStatus_t::UInt32 begin
    CUSPARSE_STATUS_SUCCESS = 0
    CUSPARSE_STATUS_NOT_INITIALIZED = 1
    CUSPARSE_STATUS_ALLOC_FAILED = 2
    CUSPARSE_STATUS_INVALID_VALUE = 3
    CUSPARSE_STATUS_ARCH_MISMATCH = 4
    CUSPARSE_STATUS_MAPPING_ERROR = 5
    CUSPARSE_STATUS_EXECUTION_FAILED = 6
    CUSPARSE_STATUS_INTERNAL_ERROR = 7
    CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8
    CUSPARSE_STATUS_ZERO_PIVOT = 9
    CUSPARSE_STATUS_NOT_SUPPORTED = 10
    CUSPARSE_STATUS_INSUFFICIENT_RESOURCES = 11
end

@cenum cusparsePointerMode_t::UInt32 begin
    CUSPARSE_POINTER_MODE_HOST = 0
    CUSPARSE_POINTER_MODE_DEVICE = 1
end

@cenum cusparseAction_t::UInt32 begin
    CUSPARSE_ACTION_SYMBOLIC = 0
    CUSPARSE_ACTION_NUMERIC = 1
end

@cenum cusparseMatrixType_t::UInt32 begin
    CUSPARSE_MATRIX_TYPE_GENERAL = 0
    CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1
    CUSPARSE_MATRIX_TYPE_HERMITIAN = 2
    CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3
end

@cenum cusparseFillMode_t::UInt32 begin
    CUSPARSE_FILL_MODE_LOWER = 0
    CUSPARSE_FILL_MODE_UPPER = 1
end

@cenum cusparseDiagType_t::UInt32 begin
    CUSPARSE_DIAG_TYPE_NON_UNIT = 0
    CUSPARSE_DIAG_TYPE_UNIT = 1
end

@cenum cusparseIndexBase_t::UInt32 begin
    CUSPARSE_INDEX_BASE_ZERO = 0
    CUSPARSE_INDEX_BASE_ONE = 1
end

@cenum cusparseOperation_t::UInt32 begin
    CUSPARSE_OPERATION_NON_TRANSPOSE = 0
    CUSPARSE_OPERATION_TRANSPOSE = 1
    CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
end

@cenum cusparseDirection_t::UInt32 begin
    CUSPARSE_DIRECTION_ROW = 0
    CUSPARSE_DIRECTION_COLUMN = 1
end

@cenum cusparseSolvePolicy_t::UInt32 begin
    CUSPARSE_SOLVE_POLICY_NO_LEVEL = 0
    CUSPARSE_SOLVE_POLICY_USE_LEVEL = 1
end

@cenum cusparseSideMode_t::UInt32 begin
    CUSPARSE_SIDE_LEFT = 0
    CUSPARSE_SIDE_RIGHT = 1
end

@cenum cusparseColorAlg_t::UInt32 begin
    CUSPARSE_COLOR_ALG0 = 0
    CUSPARSE_COLOR_ALG1 = 1
end

@cenum cusparseAlgMode_t::UInt32 begin
    CUSPARSE_ALG_MERGE_PATH = 0
end

@cenum cusparseCsr2CscAlg_t::UInt32 begin
    CUSPARSE_CSR2CSC_ALG1 = 1
    CUSPARSE_CSR2CSC_ALG2 = 2
end

@cenum cusparseFormat_t::UInt32 begin
    CUSPARSE_FORMAT_CSR = 1
    CUSPARSE_FORMAT_CSC = 2
    CUSPARSE_FORMAT_COO = 3
    CUSPARSE_FORMAT_COO_AOS = 4
    CUSPARSE_FORMAT_BLOCKED_ELL = 5
end

@cenum cusparseOrder_t::UInt32 begin
    CUSPARSE_ORDER_COL = 1
    CUSPARSE_ORDER_ROW = 2
end

@cenum cusparseIndexType_t::UInt32 begin
    CUSPARSE_INDEX_16U = 1
    CUSPARSE_INDEX_32I = 2
    CUSPARSE_INDEX_64I = 3
end

const cusparseSpVecDescr = Cvoid
const cusparseDnVecDescr = Cvoid
const cusparseSpMatDescr = Cvoid
const cusparseDnMatDescr = Cvoid
const cusparseSpVecDescr_t = Ptr{cusparseSpVecDescr}
const cusparseDnVecDescr_t = Ptr{cusparseDnVecDescr}
const cusparseSpMatDescr_t = Ptr{cusparseSpMatDescr}
const cusparseDnMatDescr_t = Ptr{cusparseDnMatDescr}

@cenum cusparseSparseToDenseAlg_t::UInt32 begin
    CUSPARSE_SPARSETODENSE_ALG_DEFAULT = 0
end

@cenum cusparseDenseToSparseAlg_t::UInt32 begin
    CUSPARSE_DENSETOSPARSE_ALG_DEFAULT = 0
end

@cenum cusparseSpMVAlg_t::UInt32 begin
    CUSPARSE_MV_ALG_DEFAULT = 0
    CUSPARSE_COOMV_ALG = 1
    CUSPARSE_CSRMV_ALG1 = 2
    CUSPARSE_CSRMV_ALG2 = 3
    CUSPARSE_SPMV_ALG_DEFAULT = 0
    CUSPARSE_SPMV_CSR_ALG1 = 2
    CUSPARSE_SPMV_CSR_ALG2 = 3
    CUSPARSE_SPMV_COO_ALG1 = 1
    CUSPARSE_SPMV_COO_ALG2 = 4
end

@cenum cusparseSpMMAlg_t::UInt32 begin
    CUSPARSE_MM_ALG_DEFAULT = 0
    CUSPARSE_COOMM_ALG1 = 1
    CUSPARSE_COOMM_ALG2 = 2
    CUSPARSE_COOMM_ALG3 = 3
    CUSPARSE_CSRMM_ALG1 = 4
    CUSPARSE_SPMM_ALG_DEFAULT = 0
    CUSPARSE_SPMM_COO_ALG1 = 1
    CUSPARSE_SPMM_COO_ALG2 = 2
    CUSPARSE_SPMM_COO_ALG3 = 3
    CUSPARSE_SPMM_COO_ALG4 = 5
    CUSPARSE_SPMM_CSR_ALG1 = 4
    CUSPARSE_SPMM_CSR_ALG2 = 6
    CUSPARSE_SPMM_CSR_ALG3 = 12
    CUSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13
end

@cenum cusparseSpGEMMAlg_t::UInt32 begin
    CUSPARSE_SPGEMM_DEFAULT = 0
end

const cusparseSpGEMMDescr = Cvoid
const cusparseSpGEMMDescr_t = Ptr{cusparseSpGEMMDescr}

@cenum cusparseSDDMMAlg_t::UInt32 begin
    CUSPARSE_SDDMM_ALG_DEFAULT = 0
end

