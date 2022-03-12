# Automatically generated using Clang.jl

const CUSTATEVEC_VER_MAJOR = 0
const CUSTATEVEC_VER_MINOR = 1
const CUSTATEVEC_VER_PATCH = 0
const CUSTATEVEC_VERSION = CUSTATEVEC_VER_MAJOR * 1000 + CUSTATEVEC_VER_MINOR * 100 + CUSTATEVEC_VER_PATCH
const custatevecIndex_t = Int64
const custatevecContext = Cvoid
const custatevecHandle_t = Ptr{custatevecContext}

struct custatevecSamplerDescriptor
    data::NTuple{256, UInt8}
end

const custatevecSamplerDescriptor_t = custatevecSamplerDescriptor

struct custatevecAccessorDescriptor
    data::NTuple{1024, UInt8}
end

const custatevecAccessorDescriptor_t = custatevecAccessorDescriptor
const custatevecLoggerCallback_t = Ptr{Cvoid}

@cenum custatevecStatus_t::UInt32 begin
    CUSTATEVEC_STATUS_SUCCESS = 0
    CUSTATEVEC_STATUS_NOT_INITIALIZED = 1
    CUSTATEVEC_STATUS_ALLOC_FAILED = 2
    CUSTATEVEC_STATUS_INVALID_VALUE = 3
    CUSTATEVEC_STATUS_ARCH_MISMATCH = 4
    CUSTATEVEC_STATUS_EXECUTION_FAILED = 5
    CUSTATEVEC_STATUS_INTERNAL_ERROR = 6
    CUSTATEVEC_STATUS_NOT_SUPPORTED = 7
    CUSTATEVEC_STATUS_INSUFFICIENT_WORKSPACE = 8
    CUSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED = 9
end

@cenum custatevecPauli_t::UInt32 begin
    CUSTATEVEC_PAULI_I = 0
    CUSTATEVEC_PAULI_X = 1
    CUSTATEVEC_PAULI_Y = 2
    CUSTATEVEC_PAULI_Z = 3
end

@cenum custatevecMatrixLayout_t::UInt32 begin
    CUSTATEVEC_MATRIX_LAYOUT_COL = 0
    CUSTATEVEC_MATRIX_LAYOUT_ROW = 1
end

@cenum custatevecMatrixType_t::UInt32 begin
    CUSTATEVEC_MATRIX_TYPE_GENERAL = 0
    CUSTATEVEC_MATRIX_TYPE_UNITARY = 1
    CUSTATEVEC_MATRIX_TYPE_HERMITIAN = 2
end

@cenum custatevecCollapseOp_t::UInt32 begin
    CUSTATEVEC_COLLAPSE_NONE = 0
    CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO = 1
end

@cenum custatevecComputeType_t::UInt32 begin
    CUSTATEVEC_COMPUTE_DEFAULT = 0
    CUSTATEVEC_COMPUTE_32F = 4
    CUSTATEVEC_COMPUTE_64F = 16
    CUSTATEVEC_COMPUTE_TF32 = 4096
end

@cenum custatevecSamplerOutput_t::UInt32 begin
    CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER = 0
    CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER = 1
end

