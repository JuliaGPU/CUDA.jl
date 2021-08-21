# Automatically generated using Clang.jl

# Skipping MacroDefinition: CUFFTAPI __attribute__ ( ( visibility ( "default" ) ) )

const CUFFT_VER_MAJOR = 10
const CUFFT_VER_MINOR = 4
const CUFFT_VER_PATCH = 1
const CUFFT_VER_BUILD = 152
const CUFFT_VERSION = 10401
const MAX_CUFFT_ERROR = 0x11
const CUFFT_FORWARD = -1
const CUFFT_INVERSE = 1

@cenum cufftCompatibility_t::UInt32 begin
    CUFFT_COMPATIBILITY_FFTW_PADDING = 1
end

const CUFFT_COMPATIBILITY_DEFAULT = CUFFT_COMPATIBILITY_FFTW_PADDING
const MAX_SHIM_RANK = 3

@cenum cufftResult_t::UInt32 begin
    CUFFT_SUCCESS = 0
    CUFFT_INVALID_PLAN = 1
    CUFFT_ALLOC_FAILED = 2
    CUFFT_INVALID_TYPE = 3
    CUFFT_INVALID_VALUE = 4
    CUFFT_INTERNAL_ERROR = 5
    CUFFT_EXEC_FAILED = 6
    CUFFT_SETUP_FAILED = 7
    CUFFT_INVALID_SIZE = 8
    CUFFT_UNALIGNED_DATA = 9
    CUFFT_INCOMPLETE_PARAMETER_LIST = 10
    CUFFT_INVALID_DEVICE = 11
    CUFFT_PARSE_ERROR = 12
    CUFFT_NO_WORKSPACE = 13
    CUFFT_NOT_IMPLEMENTED = 14
    CUFFT_LICENSE_ERROR = 15
    CUFFT_NOT_SUPPORTED = 16
end

const cufftResult = cufftResult_t
const cufftReal = Cfloat
const cufftDoubleReal = Cdouble
const cufftComplex = cuComplex
const cufftDoubleComplex = cuDoubleComplex

@cenum cufftType_t::UInt32 begin
    CUFFT_R2C = 42
    CUFFT_C2R = 44
    CUFFT_C2C = 41
    CUFFT_D2Z = 106
    CUFFT_Z2D = 108
    CUFFT_Z2Z = 105
end

const cufftType = cufftType_t
const cufftCompatibility = cufftCompatibility_t
const cufftHandle = Cint
