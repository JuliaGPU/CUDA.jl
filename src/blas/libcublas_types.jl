# libcublas_types.jl
#
# Initially generated with wrap_c from Clang.jl. Modified to remove anonymous
# enums and add cublasContext.
#
# Author: Nick Henderson <nwh@stanford.edu>
# Created: 2014-08-27
# License: MIT
#

# begin enum cublasStatus_t
const cublasStatus_t = UInt32
const CUBLAS_STATUS_SUCCESS = 0
const CUBLAS_STATUS_NOT_INITIALIZED = 1
const CUBLAS_STATUS_ALLOC_FAILED = 3
const CUBLAS_STATUS_INVALID_VALUE = 7
const CUBLAS_STATUS_ARCH_MISMATCH = 8
const CUBLAS_STATUS_MAPPING_ERROR = 11
const CUBLAS_STATUS_EXECUTION_FAILED = 13
const CUBLAS_STATUS_INTERNAL_ERROR = 14
const CUBLAS_STATUS_NOT_SUPPORTED = 15
const CUBLAS_STATUS_LICENSE_ERROR = 16
# end enum cublasStatus_t
# begin enum cublasFillMode_t
const cublasFillMode_t = UInt32
const CUBLAS_FILL_MODE_LOWER = 0
const CUBLAS_FILL_MODE_UPPER = 1
# end enum cublasFillMode_t
# begin enum cublasDiagType_t
const cublasDiagType_t = UInt32
const CUBLAS_DIAG_NON_UNIT = 0
const CUBLAS_DIAG_UNIT = 1
# end enum cublasDiagType_t
# begin enum cublasSideMode_t
const cublasSideMode_t = UInt32
const CUBLAS_SIDE_LEFT = 0
const CUBLAS_SIDE_RIGHT = 1
# end enum cublasSideMode_t
# begin enum cublasOperation_t
const cublasOperation_t = UInt32
const CUBLAS_OP_N = 0
const CUBLAS_OP_T = 1
const CUBLAS_OP_C = 2
# end enum cublasOperation_t
# begin enum cublasPointerMode_t
const cublasPointerMode_t = UInt32
const CUBLAS_POINTER_MODE_HOST = 0
const CUBLAS_POINTER_MODE_DEVICE = 1
# end enum cublasPointerMode_t
# begin enum cublasAtomicsMode_t
const cublasAtomicsMode_t = UInt32
const CUBLAS_ATOMICS_NOT_ALLOWED = 0
const CUBLAS_ATOMICS_ALLOWED = 1
# end enum cublasAtomicsMode_t
const cublasContext = Nothing
const cublasHandle_t = Ptr{cublasContext}
const cublasXtHandle_t = Ptr{cublasContext}
# complex numbers in cuda
const cuComplex = Complex{Float32}
const cuDoubleComplex = Complex{Float64}
# complex types from Base/linalg.jl
const CublasFloat = Union{Float64,Float32,ComplexF64,ComplexF32}
const CublasReal = Union{Float64,Float32}
const CublasComplex = Union{ComplexF64,ComplexF32}
# FP16 (cuda_fp16.h) in cuda
const __half = Float16
struct __half2
    x1::__half
    x2::__half
end

const cublasXtOpType_t = UInt32
const CUBLASXT_FLOAT = 0
const CUBLASXT_DOUBLE = 1 
const CUBLASXT_COMPLEX = 2
const CUBLASXT_DOUBLECOMPLEX = 3

const cublasXtBlasOp_t = UInt32
const CUBLASXT_GEMM = 0
const CUBLASXT_SYRK = 1 
const CUBLASXT_HERK = 2
const CUBLASXT_SYMM= 3
const CUBLASXT_HEMM= 4 
const CUBLASXT_TRSM= 5
const CUBLASXT_SYR2K= 6
const CUBLASXT_HER2K= 7
const CUBLASXT_SPMM= 8
const CUBLASXT_SYRKX= 9
const CUBLASXT_HERKX= 10

const cublasXtPinningMemMode_t = UInt32
const CUBLASXT_PINNING_DISABLED = 0
const CUBLASXT_PINNING_ENABLED = 1 

if CUDAdrv.version() >= v"0.7.5"
    # specify which GEMM algorithm to use in cublasGemmEx() (CUDA 7.5+)
    const cublasGemmAlgo_t = Int32
    const CUBLAS_GEMM_DFALT = -1
    const CUBLAS_GEMM_ALGO0 = 0
    const CUBLAS_GEMM_ALGO1 = 1
    const CUBLAS_GEMM_ALGO2 = 2
    const CUBLAS_GEMM_ALGO3 = 3
    const CUBLAS_GEMM_ALGO4 = 4
    const CUBLAS_GEMM_ALGO5 = 5
    const CUBLAS_GEMM_ALGO6 = 6
    const CUBLAS_GEMM_ALGO7 = 7
    # specify which DataType to use with cublas<t>gemmEx() and cublasGemmEx() (CUDA 7.5+) functions
    const cudaDataType_t = UInt32
    const CUDA_R_16F = UInt32(2)
    const CUDA_C_16F = UInt32(6)
    const CUDA_R_32F = UInt32(0)
    const CUDA_C_32F = UInt32(4)
    const CUDA_R_64F = UInt32(1)
    const CUDA_C_64F = UInt32(5)
    const CUDA_R_8I  = UInt32(3)
    const CUDA_C_8I  = UInt32(7)
    const CUDA_R_8U  = UInt32(8)
    const CUDA_C_8U  = UInt32(9)
    const CUDA_R_32I = UInt32(10)
    const CUDA_C_32I = UInt32(11)
    const CUDA_R_32U = UInt32(12)
    const CUDA_C_32U = UInt32(13)
end

@enum CUBLASMathMode::Cint begin
   CUBLAS_DEFAULT_MATH = 0
   CUBLAS_TENSOR_OP_MATH = 1
end
