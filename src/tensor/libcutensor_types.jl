"""
Status messages from CUTENSOR's C API.
"""
# begin enum cutensorStatus_t
const cutensorStatus_t = UInt32
const CUTENSOR_STATUS_SUCCESS                = UInt32(0)
const CUTENSOR_STATUS_NOT_INITIALIZED        = UInt32(1)
const CUTENSOR_STATUS_ALLOC_FAILED           = UInt32(3)
const CUTENSOR_STATUS_INVALID_VALUE          = UInt32(7)
const CUTENSOR_STATUS_ARCH_MISMATCH          = UInt32(8)
const CUTENSOR_STATUS_MAPPING_ERROR          = UInt32(11)
const CUTENSOR_STATUS_EXECUTION_FAILED       = UInt32(13)
const CUTENSOR_STATUS_INTERNAL_ERROR         = UInt32(14)
const CUTENSOR_STATUS_NOT_SUPPORTED          = UInt32(15)
const CUTENSOR_STATUS_LICENSE_ERROR          = UInt32(16)
const CUTENSOR_STATUS_CUBLAS_ERROR           = UInt32(17)
const CUTENSOR_STATUS_CUDA_ERROR             = UInt32(18)
const CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE = UInt32(19)
const CUTENSOR_STATUS_INSUFFICIENT_DRIVER    = UInt32(20)
# end enum cutensorStatus_t

# begin enum cutensorOperator_t
const cutensorOperator_t = UInt32
# Unary
const CUTENSOR_OP_IDENTITY = UInt32(1)
const CUTENSOR_OP_SQRT     = UInt32(2)
const CUTENSOR_OP_RELU     = UInt32(8)
const CUTENSOR_OP_CONJ     = UInt32(9)
const CUTENSOR_OP_RCP      = UInt32(10)
# Binary
const CUTENSOR_OP_ADD      = UInt32(3)
const CUTENSOR_OP_MUL      = UInt32(5)
const CUTENSOR_OP_MAX      = UInt32(6)
const CUTENSOR_OP_MIN      = UInt32(7)
const CUTENSOR_OP_UNKNOWN  = UInt32(126)
# end enum cutensorOperator_t

# begin enum cutensorWorksizePreference_t
const cutensorWorksizePreference_t = UInt32
const CUTENSOR_WORKSPACE_MIN         = UInt32(1)
const CUTENSOR_WORKSPACE_RECOMMENDED = UInt32(2)
const CUTENSOR_WORKSPACE_MAX         = UInt32(3)
# end enum cutensorWorksizePreference_t

# begin enum cutensorPaddingType_t
const cutensorPaddingType_t = UInt32
const CUTENSOR_PADDING_NONE = UInt32(0)
const CUTENSOR_PADDING_ZERO = UInt32(1)
# end enum cutensorPaddingType_t

# begin enum cutensorAlgo_t
const cutensorAlgo_t = Int32
const CUTENSOR_ALGO_TGETT          = Int32(-7)
const CUTENSOR_ALGO_GETT           = Int32(-6)
const CUTENSOR_ALGO_LOG_TENSOR_OP  = Int32(-5)
const CUTENSOR_ALGO_LOG            = Int32(-4)
const CUTENSOR_ALGO_TTGT_TENSOR_OP = Int32(-3)
const CUTENSOR_ALGO_TTGT           = Int32(-2)
const CUTENSOR_ALGO_DEFAULT        = Int32(-1)
# end enum cutensorAlgo_t

const cutensorTensorDescriptor_t = Ptr{Cvoid}
const cutensorContext = Cvoid
const cutensorHandle_t = Ptr{cutensorContext}
