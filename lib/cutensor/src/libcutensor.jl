using CEnum

# cuTENSOR uses CUDA runtime objects, which are compatible with our driver usage
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
        res = @retry_reclaim err -> $check $(esc(ex))
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
    fields::NTuple{512,Int64}
end

struct cutensorPlanCacheline_t
    fields::NTuple{1408,Int64}
end

struct cutensorPlanCache_t
    fields::NTuple{12288,Int64}
end

struct cutensorTensorDescriptor_t
    fields::NTuple{72,Int64}
end

struct cutensorContractionDescriptor_t
    fields::NTuple{288,Int64}
end

struct cutensorContractionPlan_t
    fields::NTuple{1408,Int64}
end

struct cutensorContractionFind_t
    fields::NTuple{64,Int64}
end

# typedef void ( * cutensorLoggerCallback_t ) ( int32_t logLevel , const char * functionName , const char * message )
const cutensorLoggerCallback_t = Ptr{Cvoid}

@checked function cutensorInit(handle)
    initialize_context()
    @ccall libcutensor.cutensorInit(handle::Ptr{cutensorHandle_t})::cutensorStatus_t
end

@checked function cutensorHandleDetachPlanCachelines(handle)
    initialize_context()
    @ccall libcutensor.cutensorHandleDetachPlanCachelines(handle::Ptr{cutensorHandle_t})::cutensorStatus_t
end

@checked function cutensorHandleAttachPlanCachelines(handle, cachelines, numCachelines)
    initialize_context()
    @ccall libcutensor.cutensorHandleAttachPlanCachelines(handle::Ptr{cutensorHandle_t},
                                                          cachelines::Ptr{cutensorPlanCacheline_t},
                                                          numCachelines::UInt32)::cutensorStatus_t
end

@checked function cutensorHandleWriteCacheToFile(handle, filename)
    initialize_context()
    @ccall libcutensor.cutensorHandleWriteCacheToFile(handle::Ptr{cutensorHandle_t},
                                                      filename::Ptr{Cchar})::cutensorStatus_t
end

@checked function cutensorHandleReadCacheFromFile(handle, filename, numCachelinesRead)
    initialize_context()
    @ccall libcutensor.cutensorHandleReadCacheFromFile(handle::Ptr{cutensorHandle_t},
                                                       filename::Ptr{Cchar},
                                                       numCachelinesRead::Ref{UInt32})::cutensorStatus_t
end

@checked function cutensorInitTensorDescriptor(handle, desc, numModes, extent, stride,
                                               dataType, unaryOp)
    initialize_context()
    @ccall libcutensor.cutensorInitTensorDescriptor(handle::Ptr{cutensorHandle_t},
                                                    desc::Ptr{cutensorTensorDescriptor_t},
                                                    numModes::UInt32, extent::Ptr{Int64},
                                                    stride::Ptr{Int64},
                                                    dataType::cudaDataType_t,
                                                    unaryOp::cutensorOperator_t)::cutensorStatus_t
end

@checked function cutensorElementwiseTrinary(handle, alpha, A, descA, modeA, beta, B, descB,
                                             modeB, gamma, C, descC, modeC, D, descD, modeD,
                                             opAB, opABC, typeScalar, stream)
    initialize_context()
    @ccall libcutensor.cutensorElementwiseTrinary(handle::Ptr{cutensorHandle_t},
                                                  alpha::Ptr{Cvoid}, A::PtrOrCuPtr{Cvoid},
                                                  descA::Ptr{cutensorTensorDescriptor_t},
                                                  modeA::Ptr{Int32}, beta::Ptr{Cvoid},
                                                  B::PtrOrCuPtr{Cvoid},
                                                  descB::Ptr{cutensorTensorDescriptor_t},
                                                  modeB::Ptr{Int32}, gamma::Ptr{Cvoid},
                                                  C::PtrOrCuPtr{Cvoid},
                                                  descC::Ptr{cutensorTensorDescriptor_t},
                                                  modeC::Ptr{Int32}, D::PtrOrCuPtr{Cvoid},
                                                  descD::Ptr{cutensorTensorDescriptor_t},
                                                  modeD::Ptr{Int32},
                                                  opAB::cutensorOperator_t,
                                                  opABC::cutensorOperator_t,
                                                  typeScalar::cudaDataType_t,
                                                  stream::cudaStream_t)::cutensorStatus_t
end

@checked function cutensorElementwiseBinary(handle, alpha, A, descA, modeA, gamma, C, descC,
                                            modeC, D, descD, modeD, opAC, typeScalar,
                                            stream)
    initialize_context()
    @ccall libcutensor.cutensorElementwiseBinary(handle::Ptr{cutensorHandle_t},
                                                 alpha::Ptr{Cvoid}, A::PtrOrCuPtr{Cvoid},
                                                 descA::Ptr{cutensorTensorDescriptor_t},
                                                 modeA::Ptr{Int32}, gamma::Ptr{Cvoid},
                                                 C::PtrOrCuPtr{Cvoid},
                                                 descC::Ptr{cutensorTensorDescriptor_t},
                                                 modeC::Ptr{Int32}, D::PtrOrCuPtr{Cvoid},
                                                 descD::Ptr{cutensorTensorDescriptor_t},
                                                 modeD::Ptr{Int32},
                                                 opAC::cutensorOperator_t,
                                                 typeScalar::cudaDataType_t,
                                                 stream::cudaStream_t)::cutensorStatus_t
end

@checked function cutensorPermutation(handle, alpha, A, descA, modeA, B, descB, modeB,
                                      typeScalar, stream)
    initialize_context()
    @ccall libcutensor.cutensorPermutation(handle::Ptr{cutensorHandle_t}, alpha::Ptr{Cvoid},
                                           A::PtrOrCuPtr{Cvoid},
                                           descA::Ptr{cutensorTensorDescriptor_t},
                                           modeA::Ptr{Int32}, B::PtrOrCuPtr{Cvoid},
                                           descB::Ptr{cutensorTensorDescriptor_t},
                                           modeB::Ptr{Int32}, typeScalar::cudaDataType_t,
                                           stream::cudaStream_t)::cutensorStatus_t
end

@checked function cutensorInitContractionDescriptor(handle, desc, descA, modeA,
                                                    alignmentRequirementA, descB, modeB,
                                                    alignmentRequirementB, descC, modeC,
                                                    alignmentRequirementC, descD, modeD,
                                                    alignmentRequirementD, typeCompute)
    initialize_context()
    @ccall libcutensor.cutensorInitContractionDescriptor(handle::Ptr{cutensorHandle_t},
                                                         desc::Ptr{cutensorContractionDescriptor_t},
                                                         descA::Ptr{cutensorTensorDescriptor_t},
                                                         modeA::Ptr{Int32},
                                                         alignmentRequirementA::UInt32,
                                                         descB::Ptr{cutensorTensorDescriptor_t},
                                                         modeB::Ptr{Int32},
                                                         alignmentRequirementB::UInt32,
                                                         descC::Ptr{cutensorTensorDescriptor_t},
                                                         modeC::Ptr{Int32},
                                                         alignmentRequirementC::UInt32,
                                                         descD::Ptr{cutensorTensorDescriptor_t},
                                                         modeD::Ptr{Int32},
                                                         alignmentRequirementD::UInt32,
                                                         typeCompute::cutensorComputeType_t)::cutensorStatus_t
end

@checked function cutensorContractionDescriptorSetAttribute(handle, desc, attr, buf,
                                                            sizeInBytes)
    initialize_context()
    @ccall libcutensor.cutensorContractionDescriptorSetAttribute(handle::Ptr{cutensorHandle_t},
                                                                 desc::Ptr{cutensorContractionDescriptor_t},
                                                                 attr::cutensorContractionDescriptorAttributes_t,
                                                                 buf::Ptr{Cvoid},
                                                                 sizeInBytes::Csize_t)::cutensorStatus_t
end

@checked function cutensorInitContractionFind(handle, find, algo)
    initialize_context()
    @ccall libcutensor.cutensorInitContractionFind(handle::Ptr{cutensorHandle_t},
                                                   find::Ptr{cutensorContractionFind_t},
                                                   algo::cutensorAlgo_t)::cutensorStatus_t
end

@checked function cutensorContractionFindSetAttribute(handle, find, attr, buf, sizeInBytes)
    initialize_context()
    @ccall libcutensor.cutensorContractionFindSetAttribute(handle::Ptr{cutensorHandle_t},
                                                           find::Ptr{cutensorContractionFind_t},
                                                           attr::cutensorContractionFindAttributes_t,
                                                           buf::Ptr{Cvoid},
                                                           sizeInBytes::Csize_t)::cutensorStatus_t
end

@checked function cutensorContractionGetWorkspaceSize(handle, desc, find, pref,
                                                      workspaceSize)
    initialize_context()
    @ccall libcutensor.cutensorContractionGetWorkspaceSize(handle::Ptr{cutensorHandle_t},
                                                           desc::Ptr{cutensorContractionDescriptor_t},
                                                           find::Ptr{cutensorContractionFind_t},
                                                           pref::cutensorWorksizePreference_t,
                                                           workspaceSize::Ptr{UInt64})::cutensorStatus_t
end

@checked function cutensorInitContractionPlan(handle, plan, desc, find, workspaceSize)
    initialize_context()
    @ccall libcutensor.cutensorInitContractionPlan(handle::Ptr{cutensorHandle_t},
                                                   plan::Ptr{cutensorContractionPlan_t},
                                                   desc::Ptr{cutensorContractionDescriptor_t},
                                                   find::Ptr{cutensorContractionFind_t},
                                                   workspaceSize::UInt64)::cutensorStatus_t
end

@checked function cutensorContraction(handle, plan, alpha, A, B, beta, C, D, workspace,
                                      workspaceSize, stream)
    initialize_context()
    @ccall libcutensor.cutensorContraction(handle::Ptr{cutensorHandle_t},
                                           plan::Ptr{cutensorContractionPlan_t},
                                           alpha::Ptr{Cvoid}, A::PtrOrCuPtr{Cvoid},
                                           B::PtrOrCuPtr{Cvoid}, beta::Ptr{Cvoid},
                                           C::PtrOrCuPtr{Cvoid}, D::PtrOrCuPtr{Cvoid},
                                           workspace::CuPtr{Cvoid}, workspaceSize::UInt64,
                                           stream::cudaStream_t)::cutensorStatus_t
end

@checked function cutensorContractionMaxAlgos(maxNumAlgos)
    initialize_context()
    @ccall libcutensor.cutensorContractionMaxAlgos(maxNumAlgos::Ptr{Int32})::cutensorStatus_t
end

@checked function cutensorReduction(handle, alpha, A, descA, modeA, beta, C, descC, modeC,
                                    D, descD, modeD, opReduce, typeCompute, workspace,
                                    workspaceSize, stream)
    initialize_context()
    @ccall libcutensor.cutensorReduction(handle::Ptr{cutensorHandle_t}, alpha::Ptr{Cvoid},
                                         A::PtrOrCuPtr{Cvoid},
                                         descA::Ptr{cutensorTensorDescriptor_t},
                                         modeA::Ptr{Int32}, beta::Ptr{Cvoid},
                                         C::PtrOrCuPtr{Cvoid},
                                         descC::Ptr{cutensorTensorDescriptor_t},
                                         modeC::Ptr{Int32}, D::PtrOrCuPtr{Cvoid},
                                         descD::Ptr{cutensorTensorDescriptor_t},
                                         modeD::Ptr{Int32}, opReduce::cutensorOperator_t,
                                         typeCompute::cutensorComputeType_t,
                                         workspace::PtrOrCuPtr{Cvoid},
                                         workspaceSize::UInt64,
                                         stream::cudaStream_t)::cutensorStatus_t
end

@checked function cutensorReductionGetWorkspaceSize(handle, A, descA, modeA, C, descC,
                                                    modeC, D, descD, modeD, opReduce,
                                                    typeCompute, workspaceSize)
    initialize_context()
    @ccall libcutensor.cutensorReductionGetWorkspaceSize(handle::Ptr{cutensorHandle_t},
                                                         A::PtrOrCuPtr{Cvoid},
                                                         descA::Ptr{cutensorTensorDescriptor_t},
                                                         modeA::Ptr{Int32},
                                                         C::PtrOrCuPtr{Cvoid},
                                                         descC::Ptr{cutensorTensorDescriptor_t},
                                                         modeC::Ptr{Int32},
                                                         D::PtrOrCuPtr{Cvoid},
                                                         descD::Ptr{cutensorTensorDescriptor_t},
                                                         modeD::Ptr{Int32},
                                                         opReduce::cutensorOperator_t,
                                                         typeCompute::cutensorComputeType_t,
                                                         workspaceSize::Ptr{UInt64})::cutensorStatus_t
end

@checked function cutensorGetAlignmentRequirement(handle, ptr, desc, alignmentRequirement)
    initialize_context()
    @ccall libcutensor.cutensorGetAlignmentRequirement(handle::Ptr{cutensorHandle_t},
                                                       ptr::PtrOrCuPtr{Cvoid},
                                                       desc::Ptr{cutensorTensorDescriptor_t},
                                                       alignmentRequirement::Ptr{UInt32})::cutensorStatus_t
end

function cutensorGetErrorString(error)
    @ccall libcutensor.cutensorGetErrorString(error::cutensorStatus_t)::Cstring
end

# no prototype is found for this function at cutensor.h:745:8, please use with caution
function cutensorGetVersion()
    @ccall libcutensor.cutensorGetVersion()::Csize_t
end

# no prototype is found for this function at cutensor.h:751:8, please use with caution
function cutensorGetCudartVersion()
    @ccall libcutensor.cutensorGetCudartVersion()::Csize_t
end

@checked function cutensorLoggerSetCallback(callback)
    initialize_context()
    @ccall libcutensor.cutensorLoggerSetCallback(callback::cutensorLoggerCallback_t)::cutensorStatus_t
end

@checked function cutensorLoggerSetFile(file)
    initialize_context()
    @ccall libcutensor.cutensorLoggerSetFile(file::Ptr{Libc.FILE})::cutensorStatus_t
end

@checked function cutensorLoggerOpenFile(logFile)
    initialize_context()
    @ccall libcutensor.cutensorLoggerOpenFile(logFile::Cstring)::cutensorStatus_t
end

@checked function cutensorLoggerSetLevel(level)
    initialize_context()
    @ccall libcutensor.cutensorLoggerSetLevel(level::Int32)::cutensorStatus_t
end

@checked function cutensorLoggerSetMask(mask)
    initialize_context()
    @ccall libcutensor.cutensorLoggerSetMask(mask::Int32)::cutensorStatus_t
end

# no prototype is found for this function at cutensor.h:799:18, please use with caution
@checked function cutensorLoggerForceDisable()
    initialize_context()
    @ccall libcutensor.cutensorLoggerForceDisable()::cutensorStatus_t
end

@checked function cutensorContractionGetWorkspace(handle, desc, find, pref, workspaceSize)
    initialize_context()
    @ccall libcutensor.cutensorContractionGetWorkspace(handle::Ptr{cutensorHandle_t},
                                                       desc::Ptr{cutensorContractionDescriptor_t},
                                                       find::Ptr{cutensorContractionFind_t},
                                                       pref::cutensorWorksizePreference_t,
                                                       workspaceSize::Ptr{UInt64})::cutensorStatus_t
end

@checked function cutensorReductionGetWorkspace(handle, A, descA, modeA, C, descC, modeC, D,
                                                descD, modeD, opReduce, typeCompute,
                                                workspaceSize)
    initialize_context()
    @ccall libcutensor.cutensorReductionGetWorkspace(handle::Ptr{cutensorHandle_t},
                                                     A::PtrOrCuPtr{Cvoid},
                                                     descA::Ptr{cutensorTensorDescriptor_t},
                                                     modeA::Ptr{Int32},
                                                     C::PtrOrCuPtr{Cvoid},
                                                     descC::Ptr{cutensorTensorDescriptor_t},
                                                     modeC::Ptr{Int32},
                                                     D::PtrOrCuPtr{Cvoid},
                                                     descD::Ptr{cutensorTensorDescriptor_t},
                                                     modeD::Ptr{Int32},
                                                     opReduce::cutensorOperator_t,
                                                     typeCompute::cutensorComputeType_t,
                                                     workspaceSize::Ptr{UInt64})::cutensorStatus_t
end
