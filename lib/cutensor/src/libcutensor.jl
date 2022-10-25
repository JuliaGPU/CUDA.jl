using CEnum

# CUTENSOR uses CUDA runtime objects, which are compatible with our driver usage
const cudaStream_t = CUstream

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == CUTENSOR_STATUS_ALLOC_FAILED
        throw(OutOfGPUMemoryError())
    else
        throw(CUTENSORError(res))
    end
end

macro check(ex, errs...)
    check = :(isequal(err, CUTENSOR_STATUS_ALLOC_FAILED))
    for err in errs
        check = :($check || isequal(err, $(esc(err))))
    end

    quote
        res = @retry_reclaim err->$check $(esc(ex))
        if res != CUTENSOR_STATUS_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end


@cenum cutensorOperator_t::UInt32 begin
    CUTENSOR_OP_IDENTITY = 1
    CUTENSOR_OP_SQRT = 2
    CUTENSOR_OP_RELU = 8
    CUTENSOR_OP_CONJ = 9
    CUTENSOR_OP_RCP = 10
    CUTENSOR_OP_SIGMOID = 11
    CUTENSOR_OP_TANH = 12
    CUTENSOR_OP_EXP = 22
    CUTENSOR_OP_LOG = 23
    CUTENSOR_OP_ABS = 24
    CUTENSOR_OP_NEG = 25
    CUTENSOR_OP_SIN = 26
    CUTENSOR_OP_COS = 27
    CUTENSOR_OP_TAN = 28
    CUTENSOR_OP_SINH = 29
    CUTENSOR_OP_COSH = 30
    CUTENSOR_OP_ASIN = 31
    CUTENSOR_OP_ACOS = 32
    CUTENSOR_OP_ATAN = 33
    CUTENSOR_OP_ASINH = 34
    CUTENSOR_OP_ACOSH = 35
    CUTENSOR_OP_ATANH = 36
    CUTENSOR_OP_CEIL = 37
    CUTENSOR_OP_FLOOR = 38
    CUTENSOR_OP_ADD = 3
    CUTENSOR_OP_MUL = 5
    CUTENSOR_OP_MAX = 6
    CUTENSOR_OP_MIN = 7
    CUTENSOR_OP_UNKNOWN = 126
end

@cenum cutensorStatus_t::UInt32 begin
    CUTENSOR_STATUS_SUCCESS = 0
    CUTENSOR_STATUS_NOT_INITIALIZED = 1
    CUTENSOR_STATUS_ALLOC_FAILED = 3
    CUTENSOR_STATUS_INVALID_VALUE = 7
    CUTENSOR_STATUS_ARCH_MISMATCH = 8
    CUTENSOR_STATUS_MAPPING_ERROR = 11
    CUTENSOR_STATUS_EXECUTION_FAILED = 13
    CUTENSOR_STATUS_INTERNAL_ERROR = 14
    CUTENSOR_STATUS_NOT_SUPPORTED = 15
    CUTENSOR_STATUS_LICENSE_ERROR = 16
    CUTENSOR_STATUS_CUBLAS_ERROR = 17
    CUTENSOR_STATUS_CUDA_ERROR = 18
    CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE = 19
    CUTENSOR_STATUS_INSUFFICIENT_DRIVER = 20
    CUTENSOR_STATUS_IO_ERROR = 21
end

@cenum cutensorAlgo_t::Int32 begin
    CUTENSOR_ALGO_DEFAULT_PATIENT = -6
    CUTENSOR_ALGO_GETT = -4
    CUTENSOR_ALGO_TGETT = -3
    CUTENSOR_ALGO_TTGT = -2
    CUTENSOR_ALGO_DEFAULT = -1
end

@cenum cutensorWorksizePreference_t::UInt32 begin
    CUTENSOR_WORKSPACE_MIN = 1
    CUTENSOR_WORKSPACE_RECOMMENDED = 2
    CUTENSOR_WORKSPACE_MAX = 3
end

@cenum cutensorComputeType_t::UInt32 begin
    CUTENSOR_COMPUTE_16F = 1
    CUTENSOR_COMPUTE_16BF = 1024
    CUTENSOR_COMPUTE_TF32 = 4096
    CUTENSOR_COMPUTE_32F = 4
    CUTENSOR_COMPUTE_64F = 16
    CUTENSOR_COMPUTE_8U = 64
    CUTENSOR_COMPUTE_8I = 256
    CUTENSOR_COMPUTE_32U = 128
    CUTENSOR_COMPUTE_32I = 512
    CUTENSOR_R_MIN_16F = 1
    CUTENSOR_C_MIN_16F = 2
    CUTENSOR_R_MIN_32F = 4
    CUTENSOR_C_MIN_32F = 8
    CUTENSOR_R_MIN_64F = 16
    CUTENSOR_C_MIN_64F = 32
    CUTENSOR_R_MIN_8U = 64
    CUTENSOR_R_MIN_32U = 128
    CUTENSOR_R_MIN_8I = 256
    CUTENSOR_R_MIN_32I = 512
    CUTENSOR_R_MIN_16BF = 1024
    CUTENSOR_R_MIN_TF32 = 2048
    CUTENSOR_C_MIN_TF32 = 4096
end

@cenum cutensorContractionDescriptorAttributes_t::UInt32 begin
    CUTENSOR_CONTRACTION_DESCRIPTOR_TAG = 0
end

@cenum cutensorContractionFindAttributes_t::UInt32 begin
    CUTENSOR_CONTRACTION_FIND_AUTOTUNE_MODE = 0
    CUTENSOR_CONTRACTION_FIND_CACHE_MODE = 1
    CUTENSOR_CONTRACTION_FIND_INCREMENTAL_COUNT = 2
end

@cenum cutensorAutotuneMode_t::UInt32 begin
    CUTENSOR_AUTOTUNE_NONE = 0
    CUTENSOR_AUTOTUNE_INCREMENTAL = 1
end

@cenum cutensorCacheMode_t::UInt32 begin
    CUTENSOR_CACHE_MODE_NONE = 0
    CUTENSOR_CACHE_MODE_PEDANTIC = 1
end

struct cutensorHandle_t
    fields::NTuple{512, Int64}
end

struct cutensorPlanCacheline_t
    fields::NTuple{1408, Int64}
end

struct cutensorPlanCache_t
    fields::NTuple{12288, Int64}
end

struct cutensorTensorDescriptor_t
    fields::NTuple{72, Int64}
end

struct cutensorContractionDescriptor_t
    fields::NTuple{288, Int64}
end

struct cutensorContractionPlan_t
    fields::NTuple{1408, Int64}
end

struct cutensorContractionFind_t
    fields::NTuple{64, Int64}
end

# typedef void ( * cutensorLoggerCallback_t ) ( int32_t logLevel , const char * functionName , const char * message )
const cutensorLoggerCallback_t = Ptr{Cvoid}

@checked function cutensorInit(handle)
        initialize_context()
        ccall((:cutensorInit, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t},), handle)
    end

@checked function cutensorHandleDetachPlanCachelines(handle)
        initialize_context()
        ccall((:cutensorHandleDetachPlanCachelines, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t},), handle)
    end

@checked function cutensorHandleAttachPlanCachelines(handle, cachelines, numCachelines)
        initialize_context()
        ccall((:cutensorHandleAttachPlanCachelines, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t}, Ptr{cutensorPlanCacheline_t}, UInt32), handle, cachelines, numCachelines)
    end

@checked function cutensorHandleWriteCacheToFile(handle, filename)
        initialize_context()
        ccall((:cutensorHandleWriteCacheToFile, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t}, Ptr{Cchar}), handle, filename)
    end

@checked function cutensorHandleReadCacheFromFile(handle, filename, numCachelinesRead)
        initialize_context()
        ccall((:cutensorHandleReadCacheFromFile, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t}, Ptr{Cchar}, Ref{UInt32}), handle, filename, numCachelinesRead)
    end

@checked function cutensorInitTensorDescriptor(handle, desc, numModes, extent, stride, dataType, unaryOp)
        initialize_context()
        ccall((:cutensorInitTensorDescriptor, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t}, Ptr{cutensorTensorDescriptor_t}, UInt32, Ptr{Int64}, Ptr{Int64}, cudaDataType_t, cutensorOperator_t), handle, desc, numModes, extent, stride, dataType, unaryOp)
    end

@checked function cutensorElementwiseTrinary(handle, alpha, A, descA, modeA, beta, B, descB, modeB, gamma, C, descC, modeC, D, descD, modeD, opAB, opABC, typeScalar, stream)
        initialize_context()
        ccall((:cutensorElementwiseTrinary, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t}, Ptr{Cvoid}, PtrOrCuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, Ptr{Cvoid}, PtrOrCuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, Ptr{Cvoid}, PtrOrCuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, PtrOrCuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, cutensorOperator_t, cutensorOperator_t, cudaDataType_t, cudaStream_t), handle, alpha, A, descA, modeA, beta, B, descB, modeB, gamma, C, descC, modeC, D, descD, modeD, opAB, opABC, typeScalar, stream)
    end

@checked function cutensorElementwiseBinary(handle, alpha, A, descA, modeA, gamma, C, descC, modeC, D, descD, modeD, opAC, typeScalar, stream)
        initialize_context()
        ccall((:cutensorElementwiseBinary, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t}, Ptr{Cvoid}, PtrOrCuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, Ptr{Cvoid}, PtrOrCuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, PtrOrCuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, cutensorOperator_t, cudaDataType_t, cudaStream_t), handle, alpha, A, descA, modeA, gamma, C, descC, modeC, D, descD, modeD, opAC, typeScalar, stream)
    end

@checked function cutensorPermutation(handle, alpha, A, descA, modeA, B, descB, modeB, typeScalar, stream)
        initialize_context()
        ccall((:cutensorPermutation, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t}, Ptr{Cvoid}, PtrOrCuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, PtrOrCuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, cudaDataType_t, cudaStream_t), handle, alpha, A, descA, modeA, B, descB, modeB, typeScalar, stream)
    end

@checked function cutensorInitContractionDescriptor(handle, desc, descA, modeA, alignmentRequirementA, descB, modeB, alignmentRequirementB, descC, modeC, alignmentRequirementC, descD, modeD, alignmentRequirementD, typeCompute)
        initialize_context()
        ccall((:cutensorInitContractionDescriptor, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t}, Ptr{cutensorContractionDescriptor_t}, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, UInt32, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, UInt32, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, UInt32, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, UInt32, cutensorComputeType_t), handle, desc, descA, modeA, alignmentRequirementA, descB, modeB, alignmentRequirementB, descC, modeC, alignmentRequirementC, descD, modeD, alignmentRequirementD, typeCompute)
    end

@checked function cutensorContractionDescriptorSetAttribute(handle, desc, attr, buf, sizeInBytes)
        initialize_context()
        ccall((:cutensorContractionDescriptorSetAttribute, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t}, Ptr{cutensorContractionDescriptor_t}, cutensorContractionDescriptorAttributes_t, Ptr{Cvoid}, Csize_t), handle, desc, attr, buf, sizeInBytes)
    end

@checked function cutensorInitContractionFind(handle, find, algo)
        initialize_context()
        ccall((:cutensorInitContractionFind, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t}, Ptr{cutensorContractionFind_t}, cutensorAlgo_t), handle, find, algo)
    end

@checked function cutensorContractionFindSetAttribute(handle, find, attr, buf, sizeInBytes)
        initialize_context()
        ccall((:cutensorContractionFindSetAttribute, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t}, Ptr{cutensorContractionFind_t}, cutensorContractionFindAttributes_t, Ptr{Cvoid}, Csize_t), handle, find, attr, buf, sizeInBytes)
    end

@checked function cutensorContractionGetWorkspaceSize(handle, desc, find, pref, workspaceSize)
        initialize_context()
        ccall((:cutensorContractionGetWorkspaceSize, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t}, Ptr{cutensorContractionDescriptor_t}, Ptr{cutensorContractionFind_t}, cutensorWorksizePreference_t, Ptr{UInt64}), handle, desc, find, pref, workspaceSize)
    end

@checked function cutensorInitContractionPlan(handle, plan, desc, find, workspaceSize)
        initialize_context()
        ccall((:cutensorInitContractionPlan, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t}, Ptr{cutensorContractionPlan_t}, Ptr{cutensorContractionDescriptor_t}, Ptr{cutensorContractionFind_t}, UInt64), handle, plan, desc, find, workspaceSize)
    end

@checked function cutensorContraction(handle, plan, alpha, A, B, beta, C, D, workspace, workspaceSize, stream)
        initialize_context()
        ccall((:cutensorContraction, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t}, Ptr{cutensorContractionPlan_t}, Ptr{Cvoid}, PtrOrCuPtr{Cvoid}, PtrOrCuPtr{Cvoid}, Ptr{Cvoid}, PtrOrCuPtr{Cvoid}, PtrOrCuPtr{Cvoid}, CuPtr{Cvoid}, UInt64, cudaStream_t), handle, plan, alpha, A, B, beta, C, D, workspace, workspaceSize, stream)
    end

@checked function cutensorContractionMaxAlgos(maxNumAlgos)
        initialize_context()
        ccall((:cutensorContractionMaxAlgos, libcutensor), cutensorStatus_t, (Ptr{Int32},), maxNumAlgos)
    end

@checked function cutensorReduction(handle, alpha, A, descA, modeA, beta, C, descC, modeC, D, descD, modeD, opReduce, typeCompute, workspace, workspaceSize, stream)
        initialize_context()
        ccall((:cutensorReduction, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t}, Ptr{Cvoid}, PtrOrCuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, Ptr{Cvoid}, PtrOrCuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, PtrOrCuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, cutensorOperator_t, cutensorComputeType_t, PtrOrCuPtr{Cvoid}, UInt64, cudaStream_t), handle, alpha, A, descA, modeA, beta, C, descC, modeC, D, descD, modeD, opReduce, typeCompute, workspace, workspaceSize, stream)
    end

@checked function cutensorReductionGetWorkspaceSize(handle, A, descA, modeA, C, descC, modeC, D, descD, modeD, opReduce, typeCompute, workspaceSize)
        initialize_context()
        ccall((:cutensorReductionGetWorkspaceSize, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t}, PtrOrCuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, PtrOrCuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, PtrOrCuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, cutensorOperator_t, cutensorComputeType_t, Ptr{UInt64}), handle, A, descA, modeA, C, descC, modeC, D, descD, modeD, opReduce, typeCompute, workspaceSize)
    end

@checked function cutensorGetAlignmentRequirement(handle, ptr, desc, alignmentRequirement)
        initialize_context()
        ccall((:cutensorGetAlignmentRequirement, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t}, PtrOrCuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t}, Ptr{UInt32}), handle, ptr, desc, alignmentRequirement)
    end

function cutensorGetErrorString(error)
    ccall((:cutensorGetErrorString, libcutensor), Cstring, (cutensorStatus_t,), error)
end

# no prototype is found for this function at cutensor.h:745:8, please use with caution
function cutensorGetVersion()
    ccall((:cutensorGetVersion, libcutensor), Csize_t, ())
end

# no prototype is found for this function at cutensor.h:751:8, please use with caution
function cutensorGetCudartVersion()
    ccall((:cutensorGetCudartVersion, libcutensor), Csize_t, ())
end

@checked function cutensorLoggerSetCallback(callback)
        initialize_context()
        ccall((:cutensorLoggerSetCallback, libcutensor), cutensorStatus_t, (cutensorLoggerCallback_t,), callback)
    end

@checked function cutensorLoggerSetFile(file)
        initialize_context()
        ccall((:cutensorLoggerSetFile, libcutensor), cutensorStatus_t, (Ptr{Libc.FILE},), file)
    end

@checked function cutensorLoggerOpenFile(logFile)
        initialize_context()
        ccall((:cutensorLoggerOpenFile, libcutensor), cutensorStatus_t, (Cstring,), logFile)
    end

@checked function cutensorLoggerSetLevel(level)
        initialize_context()
        ccall((:cutensorLoggerSetLevel, libcutensor), cutensorStatus_t, (Int32,), level)
    end

@checked function cutensorLoggerSetMask(mask)
        initialize_context()
        ccall((:cutensorLoggerSetMask, libcutensor), cutensorStatus_t, (Int32,), mask)
    end

# no prototype is found for this function at cutensor.h:799:18, please use with caution
@checked function cutensorLoggerForceDisable()
        initialize_context()
        ccall((:cutensorLoggerForceDisable, libcutensor), cutensorStatus_t, ())
    end

@checked function cutensorContractionGetWorkspace(handle, desc, find, pref, workspaceSize)
        initialize_context()
        ccall((:cutensorContractionGetWorkspace, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t}, Ptr{cutensorContractionDescriptor_t}, Ptr{cutensorContractionFind_t}, cutensorWorksizePreference_t, Ptr{UInt64}), handle, desc, find, pref, workspaceSize)
    end

@checked function cutensorReductionGetWorkspace(handle, A, descA, modeA, C, descC, modeC, D, descD, modeD, opReduce, typeCompute, workspaceSize)
        initialize_context()
        ccall((:cutensorReductionGetWorkspace, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t}, PtrOrCuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, PtrOrCuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, PtrOrCuPtr{Cvoid}, Ptr{cutensorTensorDescriptor_t}, Ptr{Int32}, cutensorOperator_t, cutensorComputeType_t, Ptr{UInt64}), handle, A, descA, modeA, C, descC, modeC, D, descD, modeD, opReduce, typeCompute, workspaceSize)
    end

