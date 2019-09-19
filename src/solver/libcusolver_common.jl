# Automatically generated using Clang.jl


const CUSOLVER_VER_MAJOR = 10
const CUSOLVER_VER_MINOR = 2
const CUSOLVER_VER_PATCH = 0
const CUSOLVER_VER_BUILD = 243
const CUSOLVER_VERSION = CUSOLVER_VER_MAJOR * 1000 + CUSOLVER_VER_MINOR * 100 + CUSOLVER_VER_PATCH

@cenum cusolverStatus_t::UInt32 begin
    CUSOLVER_STATUS_SUCCESS = 0
    CUSOLVER_STATUS_NOT_INITIALIZED = 1
    CUSOLVER_STATUS_ALLOC_FAILED = 2
    CUSOLVER_STATUS_INVALID_VALUE = 3
    CUSOLVER_STATUS_ARCH_MISMATCH = 4
    CUSOLVER_STATUS_MAPPING_ERROR = 5
    CUSOLVER_STATUS_EXECUTION_FAILED = 6
    CUSOLVER_STATUS_INTERNAL_ERROR = 7
    CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8
    CUSOLVER_STATUS_NOT_SUPPORTED = 9
    CUSOLVER_STATUS_ZERO_PIVOT = 10
    CUSOLVER_STATUS_INVALID_LICENSE = 11
end

@cenum cusolverEigType_t::UInt32 begin
    CUSOLVER_EIG_TYPE_1 = 1
    CUSOLVER_EIG_TYPE_2 = 2
    CUSOLVER_EIG_TYPE_3 = 3
end

@cenum cusolverEigMode_t::UInt32 begin
    CUSOLVER_EIG_MODE_NOVECTOR = 0
    CUSOLVER_EIG_MODE_VECTOR = 1
end

@cenum cusolverEigRange_t::UInt32 begin
    CUSOLVER_EIG_RANGE_ALL = 1001
    CUSOLVER_EIG_RANGE_I = 1002
    CUSOLVER_EIG_RANGE_V = 1003
end


const cusolverDnContext = Cvoid
const cusolverDnHandle_t = Ptr{cusolverDnContext}
const syevjInfo = Cvoid
const syevjInfo_t = Ptr{syevjInfo}
const gesvdjInfo = Cvoid
const gesvdjInfo_t = Ptr{gesvdjInfo}
const cusolverSpContext = Cvoid
const cusolverSpHandle_t = Ptr{cusolverSpContext}
const csrqrInfo = Cvoid
const csrqrInfo_t = Ptr{csrqrInfo}
