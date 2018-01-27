# CUFFT API function return values
@enum(cufftResult,
    CUFFT_SUCCESS        = 0,  #  The cuFFT operation was successful
    CUFFT_INVALID_PLAN   = 1,  #  cuFFT was passed an invalid plan handle
    CUFFT_ALLOC_FAILED   = 2,  #  cuFFT failed to allocate GPU or CPU memory
    CUFFT_INVALID_TYPE   = 3,  #  No longer used
    CUFFT_INVALID_VALUE  = 4,  #  User specified an invalid pointer or parameter
    CUFFT_INTERNAL_ERROR = 5,  #  Driver or internal cuFFT library error
    CUFFT_EXEC_FAILED    = 6,  #  Failed to execute an FFT on the GPU
    CUFFT_SETUP_FAILED   = 7,  #  The cuFFT library failed to initialize
    CUFFT_INVALID_SIZE   = 8,  #  User specified an invalid transform size
    CUFFT_UNALIGNED_DATA = 9,  #  No longer used
    CUFFT_INCOMPLETE_PARAMETER_LIST = 10, #  Missing parameters in call
    CUFFT_INVALID_DEVICE = 11, #  Execution of a plan was on different GPU than plan creation
    CUFFT_PARSE_ERROR    = 12, #  Internal plan database error
    CUFFT_NO_WORKSPACE   = 13,  #  No workspace has been provided prior to plan execution
    CUFFT_NOT_IMPLEMENTED = 14, # Function does not implement functionality for parameters given.
    CUFFT_LICENSE_ERROR  = 15, # Used in previous versions.
    CUFFT_NOT_SUPPORTED  = 16  # Operation is not supported for parameters given.
)

const cufftReal = Float32
const cufftDoubleReal = Float64

const cufftComplex = Complex64
const cufftDoubleComplex = Complex128

# CUFFT transform directions
const CUFFT_FORWARD = -1 # Forward FFT
const CUFFT_INVERSE =  1 # Inverse FFT

# CUFFT supports the following transform types
const cufftType = Cint
const CUFFT_R2C = 0x2a     # Real to Complex
const CUFFT_C2R = 0x2c     # Complex to Real
const CUFFT_C2C = 0x29     # Complex to Complex
const CUFFT_D2Z = 0x6a     # Double to Double-Complex
const CUFFT_Z2D = 0x6c     # Double-Complex to Double
const CUFFT_Z2Z = 0x69     # Double-Complex to Double-Complex

const cufftCompatibility = Cint
const   CUFFT_COMPATIBILITY_NATIVE          = 0x00
const   CUFFT_COMPATIBILITY_FFTW_PADDING    = 0x01
const   CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC = 0x02
const   CUFFT_COMPATIBILITY_FFTW_ALL        = 0x03

const cufftHandle = Cint
