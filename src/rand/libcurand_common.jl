# Automatically generated using Clang.jl


const CURAND_VER_MAJOR = 10
const CURAND_VER_MINOR = 1
const CURAND_VER_PATCH = 2
const CURAND_VER_BUILD = 89
const CURAND_VERSION = CURAND_VER_MAJOR * 1000 + CURAND_VER_MINOR * 100 + CURAND_VER_PATCH

@cenum curandStatus::UInt32 begin
    CURAND_STATUS_SUCCESS = 0
    CURAND_STATUS_VERSION_MISMATCH = 100
    CURAND_STATUS_NOT_INITIALIZED = 101
    CURAND_STATUS_ALLOCATION_FAILED = 102
    CURAND_STATUS_TYPE_ERROR = 103
    CURAND_STATUS_OUT_OF_RANGE = 104
    CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105
    CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106
    CURAND_STATUS_LAUNCH_FAILURE = 201
    CURAND_STATUS_PREEXISTING_FAILURE = 202
    CURAND_STATUS_INITIALIZATION_FAILED = 203
    CURAND_STATUS_ARCH_MISMATCH = 204
    CURAND_STATUS_INTERNAL_ERROR = 999
end


const curandStatus_t = curandStatus

@cenum curandRngType::UInt32 begin
    CURAND_RNG_TEST = 0
    CURAND_RNG_PSEUDO_DEFAULT = 100
    CURAND_RNG_PSEUDO_XORWOW = 101
    CURAND_RNG_PSEUDO_MRG32K3A = 121
    CURAND_RNG_PSEUDO_MTGP32 = 141
    CURAND_RNG_PSEUDO_MT19937 = 142
    CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161
    CURAND_RNG_QUASI_DEFAULT = 200
    CURAND_RNG_QUASI_SOBOL32 = 201
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202
    CURAND_RNG_QUASI_SOBOL64 = 203
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204
end


const curandRngType_t = curandRngType

@cenum curandOrdering::UInt32 begin
    CURAND_ORDERING_PSEUDO_BEST = 100
    CURAND_ORDERING_PSEUDO_DEFAULT = 101
    CURAND_ORDERING_PSEUDO_SEEDED = 102
    CURAND_ORDERING_QUASI_DEFAULT = 201
end


const curandOrdering_t = curandOrdering

@cenum curandDirectionVectorSet::UInt32 begin
    CURAND_DIRECTION_VECTORS_32_JOEKUO6 = 101
    CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102
    CURAND_DIRECTION_VECTORS_64_JOEKUO6 = 103
    CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104
end


const curandDirectionVectorSet_t = curandDirectionVectorSet
const curandDirectionVectors32_t = NTuple{32, UInt32}
const curandDirectionVectors64_t = NTuple{64, Culonglong}
const curandGenerator_st = Cvoid
const curandGenerator_t = Ptr{curandGenerator_st}
const curandDistribution_st = Cdouble
const curandDistribution_t = Ptr{curandDistribution_st}
const curandDistributionShift_st = Cvoid
const curandDistributionShift_t = Ptr{curandDistributionShift_st}
const curandDistributionM2Shift_st = Cvoid
const curandDistributionM2Shift_t = Ptr{curandDistributionM2Shift_st}
const curandHistogramM2_st = Cvoid
const curandHistogramM2_t = Ptr{curandHistogramM2_st}
const curandHistogramM2K_st = UInt32
const curandHistogramM2K_t = Ptr{curandHistogramM2K_st}
const curandHistogramM2V_st = curandDistribution_st
const curandHistogramM2V_t = Ptr{curandHistogramM2V_st}
const curandDiscreteDistribution_st = Cvoid
const curandDiscreteDistribution_t = Ptr{curandDiscreteDistribution_st}

@cenum curandMethod::UInt32 begin
    CURAND_CHOOSE_BEST = 0
    CURAND_ITR = 1
    CURAND_KNUTH = 2
    CURAND_HITR = 3
    CURAND_M1 = 4
    CURAND_M2 = 5
    CURAND_BINARY_SEARCH = 6
    CURAND_DISCRETE_GAUSS = 7
    CURAND_REJECTION = 8
    CURAND_DEVICE_API = 9
    CURAND_FAST_REJECTION = 10
    CURAND_3RD = 11
    CURAND_DEFINITION = 12
    CURAND_POISSON = 13
end


const curandMethod_t = curandMethod
