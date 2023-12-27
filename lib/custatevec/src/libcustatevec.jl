using CEnum

# CUSTATEVEC uses CUDA runtime objects, which are compatible with our driver usage
const cudaStream_t = CUstream

# vector types
const int2 = Tuple{Int32,Int32}

# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    if res == CUSTATEVEC_STATUS_ALLOC_FAILED
        throw(OutOfGPUMemoryError())
    else
        throw(CUSTATEVECError(res))
    end
end

function check(f, errs...)
    res = retry_reclaim(in((CUSTATEVEC_STATUS_ALLOC_FAILED, errs...))) do
        return f()
    end

    if res != CUSTATEVEC_STATUS_SUCCESS
        throw_api_error(res)
    end

    return
end

const custatevecIndex_t = Int64

mutable struct custatevecContext end

const custatevecHandle_t = Ptr{custatevecContext}

mutable struct custatevecSamplerDescriptor end

const custatevecSamplerDescriptor_t = Ptr{custatevecSamplerDescriptor}

mutable struct custatevecAccessorDescriptor end

const custatevecAccessorDescriptor_t = Ptr{custatevecAccessorDescriptor}

# typedef void ( * custatevecLoggerCallback_t ) ( int32_t logLevel , const char * functionName , const char * message )
const custatevecLoggerCallback_t = Ptr{Cvoid}

# typedef void ( * custatevecLoggerCallbackData_t ) ( int32_t logLevel , const char * functionName , const char * message , void * userData )
const custatevecLoggerCallbackData_t = Ptr{Cvoid}

struct custatevecDeviceMemHandler_t
    ctx::Ptr{Cvoid}
    device_alloc::Ptr{Cvoid}
    device_free::Ptr{Cvoid}
    name::NTuple{64,Cchar}
end

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
    CUSTATEVEC_STATUS_NO_DEVICE_ALLOCATOR = 10
    CUSTATEVEC_STATUS_DEVICE_ALLOCATOR_ERROR = 11
    CUSTATEVEC_STATUS_COMMUNICATOR_ERROR = 12
    CUSTATEVEC_STATUS_LOADING_LIBRARY_FAILED = 13
    CUSTATEVEC_STATUS_MAX_VALUE = 14
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

@cenum custatevecMatrixMapType_t::UInt32 begin
    CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST = 0
    CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED = 1
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

@cenum custatevecDeviceNetworkType_t::UInt32 begin
    CUSTATEVEC_DEVICE_NETWORK_TYPE_SWITCH = 1
    CUSTATEVEC_DEVICE_NETWORK_TYPE_FULLMESH = 2
end

@cenum custatevecStateVectorType_t::UInt32 begin
    CUSTATEVEC_STATE_VECTOR_TYPE_ZERO = 0
    CUSTATEVEC_STATE_VECTOR_TYPE_UNIFORM = 1
    CUSTATEVEC_STATE_VECTOR_TYPE_GHZ = 2
    CUSTATEVEC_STATE_VECTOR_TYPE_W = 3
end

@checked function custatevecCreate(handle)
    initialize_context()
    @ccall libcustatevec.custatevecCreate(handle::Ptr{custatevecHandle_t})::custatevecStatus_t
end

@checked function custatevecDestroy(handle)
    initialize_context()
    @ccall libcustatevec.custatevecDestroy(handle::custatevecHandle_t)::custatevecStatus_t
end

@checked function custatevecGetDefaultWorkspaceSize(handle, workspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecGetDefaultWorkspaceSize(handle::custatevecHandle_t,
                                                           workspaceSizeInBytes::Ptr{Csize_t})::custatevecStatus_t
end

@checked function custatevecSetWorkspace(handle, workspace, workspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecSetWorkspace(handle::custatevecHandle_t,
                                                workspace::CuPtr{Cvoid},
                                                workspaceSizeInBytes::Csize_t)::custatevecStatus_t
end

function custatevecGetErrorName(status)
    initialize_context()
    @ccall libcustatevec.custatevecGetErrorName(status::custatevecStatus_t)::Cstring
end

function custatevecGetErrorString(status)
    @ccall libcustatevec.custatevecGetErrorString(status::custatevecStatus_t)::Cstring
end

@checked function custatevecGetProperty(type, value)
    initialize_context()
    @ccall libcustatevec.custatevecGetProperty(type::libraryPropertyType,
                                               value::Ptr{Int32})::custatevecStatus_t
end

# no prototype is found for this function at custatevec.h:521:8, please use with caution
function custatevecGetVersion()
    @ccall libcustatevec.custatevecGetVersion()::Csize_t
end

@checked function custatevecSetStream(handle, streamId)
    initialize_context()
    @ccall libcustatevec.custatevecSetStream(handle::custatevecHandle_t,
                                             streamId::cudaStream_t)::custatevecStatus_t
end

@checked function custatevecGetStream(handle, streamId)
    initialize_context()
    @ccall libcustatevec.custatevecGetStream(handle::custatevecHandle_t,
                                             streamId::Ptr{cudaStream_t})::custatevecStatus_t
end

@checked function custatevecLoggerSetCallback(callback)
    @ccall libcustatevec.custatevecLoggerSetCallback(callback::custatevecLoggerCallback_t)::custatevecStatus_t
end

@checked function custatevecLoggerSetCallbackData(callback, userData)
    initialize_context()
    @ccall libcustatevec.custatevecLoggerSetCallbackData(callback::custatevecLoggerCallbackData_t,
                                                         userData::Ptr{Cvoid})::custatevecStatus_t
end

@checked function custatevecLoggerSetFile(file)
    @ccall libcustatevec.custatevecLoggerSetFile(file::Ptr{Libc.FILE})::custatevecStatus_t
end

@checked function custatevecLoggerOpenFile(logFile)
    @ccall libcustatevec.custatevecLoggerOpenFile(logFile::Cstring)::custatevecStatus_t
end

@checked function custatevecLoggerSetLevel(level)
    initialize_context()
    @ccall libcustatevec.custatevecLoggerSetLevel(level::Int32)::custatevecStatus_t
end

@checked function custatevecLoggerSetMask(mask)
    @ccall libcustatevec.custatevecLoggerSetMask(mask::Int32)::custatevecStatus_t
end

# no prototype is found for this function at custatevec.h:621:1, please use with caution
@checked function custatevecLoggerForceDisable()
    @ccall libcustatevec.custatevecLoggerForceDisable()::custatevecStatus_t
end

@checked function custatevecGetDeviceMemHandler(handle, handler)
    initialize_context()
    @ccall libcustatevec.custatevecGetDeviceMemHandler(handle::custatevecHandle_t,
                                                       handler::Ptr{custatevecDeviceMemHandler_t})::custatevecStatus_t
end

@checked function custatevecSetDeviceMemHandler(handle, handler)
    initialize_context()
    @ccall libcustatevec.custatevecSetDeviceMemHandler(handle::custatevecHandle_t,
                                                       handler::Ptr{custatevecDeviceMemHandler_t})::custatevecStatus_t
end

@checked function custatevecAbs2SumOnZBasis(handle, sv, svDataType, nIndexBits, abs2sum0,
                                            abs2sum1, basisBits, nBasisBits)
    initialize_context()
    @ccall libcustatevec.custatevecAbs2SumOnZBasis(handle::custatevecHandle_t,
                                                   sv::CuPtr{Cvoid},
                                                   svDataType::cudaDataType_t,
                                                   nIndexBits::UInt32,
                                                   abs2sum0::Ptr{Cdouble},
                                                   abs2sum1::Ptr{Cdouble},
                                                   basisBits::Ptr{Int32},
                                                   nBasisBits::UInt32)::custatevecStatus_t
end

@checked function custatevecAbs2SumArray(handle, sv, svDataType, nIndexBits, abs2sum,
                                         bitOrdering, bitOrderingLen, maskBitString,
                                         maskOrdering, maskLen)
    initialize_context()
    @ccall libcustatevec.custatevecAbs2SumArray(handle::custatevecHandle_t,
                                                sv::CuPtr{Cvoid},
                                                svDataType::cudaDataType_t,
                                                nIndexBits::UInt32, abs2sum::Ptr{Cdouble},
                                                bitOrdering::Ptr{Int32},
                                                bitOrderingLen::UInt32,
                                                maskBitString::Ptr{Int32},
                                                maskOrdering::Ptr{Int32},
                                                maskLen::UInt32)::custatevecStatus_t
end

@checked function custatevecCollapseOnZBasis(handle, sv, svDataType, nIndexBits, parity,
                                             basisBits, nBasisBits, norm)
    initialize_context()
    @ccall libcustatevec.custatevecCollapseOnZBasis(handle::custatevecHandle_t,
                                                    sv::CuPtr{Cvoid},
                                                    svDataType::cudaDataType_t,
                                                    nIndexBits::UInt32, parity::Int32,
                                                    basisBits::Ptr{Int32},
                                                    nBasisBits::UInt32,
                                                    norm::Cdouble)::custatevecStatus_t
end

@checked function custatevecCollapseByBitString(handle, sv, svDataType, nIndexBits,
                                                bitString, bitOrdering, bitStringLen, norm)
    initialize_context()
    @ccall libcustatevec.custatevecCollapseByBitString(handle::custatevecHandle_t,
                                                       sv::CuPtr{Cvoid},
                                                       svDataType::cudaDataType_t,
                                                       nIndexBits::UInt32,
                                                       bitString::Ptr{Int32},
                                                       bitOrdering::Ptr{Int32},
                                                       bitStringLen::UInt32,
                                                       norm::Cdouble)::custatevecStatus_t
end

@checked function custatevecMeasureOnZBasis(handle, sv, svDataType, nIndexBits, parity,
                                            basisBits, nBasisBits, randnum, collapse)
    initialize_context()
    @ccall libcustatevec.custatevecMeasureOnZBasis(handle::custatevecHandle_t,
                                                   sv::CuPtr{Cvoid},
                                                   svDataType::cudaDataType_t,
                                                   nIndexBits::UInt32, parity::Ptr{Int32},
                                                   basisBits::Ptr{Int32},
                                                   nBasisBits::UInt32, randnum::Cdouble,
                                                   collapse::custatevecCollapseOp_t)::custatevecStatus_t
end

@checked function custatevecBatchMeasure(handle, sv, svDataType, nIndexBits, bitString,
                                         bitOrdering, bitStringLen, randnum, collapse)
    initialize_context()
    @ccall libcustatevec.custatevecBatchMeasure(handle::custatevecHandle_t,
                                                sv::CuPtr{Cvoid},
                                                svDataType::cudaDataType_t,
                                                nIndexBits::UInt32, bitString::Ptr{Int32},
                                                bitOrdering::Ptr{Int32},
                                                bitStringLen::UInt32, randnum::Cdouble,
                                                collapse::custatevecCollapseOp_t)::custatevecStatus_t
end

@checked function custatevecBatchMeasureWithOffset(handle, sv, svDataType, nIndexBits,
                                                   bitString, bitOrdering, bitStringLen,
                                                   randnum, collapse, offset, abs2sum)
    initialize_context()
    @ccall libcustatevec.custatevecBatchMeasureWithOffset(handle::custatevecHandle_t,
                                                          sv::CuPtr{Cvoid},
                                                          svDataType::cudaDataType_t,
                                                          nIndexBits::UInt32,
                                                          bitString::Ptr{Int32},
                                                          bitOrdering::Ptr{Int32},
                                                          bitStringLen::UInt32,
                                                          randnum::Cdouble,
                                                          collapse::custatevecCollapseOp_t,
                                                          offset::Cdouble,
                                                          abs2sum::Cdouble)::custatevecStatus_t
end

@checked function custatevecApplyPauliRotation(handle, sv, svDataType, nIndexBits, theta,
                                               paulis, targets, nTargets, controls,
                                               controlBitValues, nControls)
    initialize_context()
    @ccall libcustatevec.custatevecApplyPauliRotation(handle::custatevecHandle_t,
                                                      sv::CuPtr{Cvoid},
                                                      svDataType::cudaDataType_t,
                                                      nIndexBits::UInt32, theta::Cdouble,
                                                      paulis::Ptr{custatevecPauli_t},
                                                      targets::Ptr{Int32}, nTargets::UInt32,
                                                      controls::Ptr{Int32},
                                                      controlBitValues::Ptr{Int32},
                                                      nControls::UInt32)::custatevecStatus_t
end

@checked function custatevecApplyMatrixGetWorkspaceSize(handle, svDataType, nIndexBits,
                                                        matrix, matrixDataType, layout,
                                                        adjoint, nTargets, nControls,
                                                        computeType,
                                                        extraWorkspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecApplyMatrixGetWorkspaceSize(handle::custatevecHandle_t,
                                                               svDataType::cudaDataType_t,
                                                               nIndexBits::UInt32,
                                                               matrix::PtrOrCuPtr{Cvoid},
                                                               matrixDataType::cudaDataType_t,
                                                               layout::custatevecMatrixLayout_t,
                                                               adjoint::Int32,
                                                               nTargets::UInt32,
                                                               nControls::UInt32,
                                                               computeType::custatevecComputeType_t,
                                                               extraWorkspaceSizeInBytes::Ptr{Csize_t})::custatevecStatus_t
end

@checked function custatevecApplyMatrix(handle, sv, svDataType, nIndexBits, matrix,
                                        matrixDataType, layout, adjoint, targets, nTargets,
                                        controls, controlBitValues, nControls, computeType,
                                        extraWorkspace, extraWorkspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecApplyMatrix(handle::custatevecHandle_t, sv::CuPtr{Cvoid},
                                               svDataType::cudaDataType_t,
                                               nIndexBits::UInt32,
                                               matrix::PtrOrCuPtr{Cvoid},
                                               matrixDataType::cudaDataType_t,
                                               layout::custatevecMatrixLayout_t,
                                               adjoint::Int32, targets::Ptr{Int32},
                                               nTargets::UInt32, controls::Ptr{Int32},
                                               controlBitValues::Ptr{Int32},
                                               nControls::UInt32,
                                               computeType::custatevecComputeType_t,
                                               extraWorkspace::CuPtr{Cvoid},
                                               extraWorkspaceSizeInBytes::Csize_t)::custatevecStatus_t
end

@checked function custatevecComputeExpectationGetWorkspaceSize(handle, svDataType,
                                                               nIndexBits, matrix,
                                                               matrixDataType, layout,
                                                               nBasisBits, computeType,
                                                               extraWorkspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecComputeExpectationGetWorkspaceSize(handle::custatevecHandle_t,
                                                                      svDataType::cudaDataType_t,
                                                                      nIndexBits::UInt32,
                                                                      matrix::PtrOrCuPtr{Cvoid},
                                                                      matrixDataType::cudaDataType_t,
                                                                      layout::custatevecMatrixLayout_t,
                                                                      nBasisBits::UInt32,
                                                                      computeType::custatevecComputeType_t,
                                                                      extraWorkspaceSizeInBytes::Ptr{Csize_t})::custatevecStatus_t
end

@checked function custatevecComputeExpectation(handle, sv, svDataType, nIndexBits,
                                               expectationValue, expectationDataType,
                                               residualNorm, matrix, matrixDataType, layout,
                                               basisBits, nBasisBits, computeType,
                                               extraWorkspace, extraWorkspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecComputeExpectation(handle::custatevecHandle_t,
                                                      sv::CuPtr{Cvoid},
                                                      svDataType::cudaDataType_t,
                                                      nIndexBits::UInt32,
                                                      expectationValue::Ptr{Cvoid},
                                                      expectationDataType::cudaDataType_t,
                                                      residualNorm::Ptr{Cdouble},
                                                      matrix::PtrOrCuPtr{Cvoid},
                                                      matrixDataType::cudaDataType_t,
                                                      layout::custatevecMatrixLayout_t,
                                                      basisBits::Ptr{Int32},
                                                      nBasisBits::UInt32,
                                                      computeType::custatevecComputeType_t,
                                                      extraWorkspace::CuPtr{Cvoid},
                                                      extraWorkspaceSizeInBytes::Csize_t)::custatevecStatus_t
end

@checked function custatevecSamplerCreate(handle, sv, svDataType, nIndexBits, sampler,
                                          nMaxShots, extraWorkspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecSamplerCreate(handle::custatevecHandle_t,
                                                 sv::CuPtr{Cvoid},
                                                 svDataType::cudaDataType_t,
                                                 nIndexBits::UInt32,
                                                 sampler::Ptr{custatevecSamplerDescriptor_t},
                                                 nMaxShots::UInt32,
                                                 extraWorkspaceSizeInBytes::Ptr{Csize_t})::custatevecStatus_t
end

@checked function custatevecSamplerDestroy(sampler)
    initialize_context()
    @ccall libcustatevec.custatevecSamplerDestroy(sampler::custatevecSamplerDescriptor_t)::custatevecStatus_t
end

@checked function custatevecSamplerPreprocess(handle, sampler, extraWorkspace,
                                              extraWorkspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecSamplerPreprocess(handle::custatevecHandle_t,
                                                     sampler::custatevecSamplerDescriptor_t,
                                                     extraWorkspace::CuPtr{Cvoid},
                                                     extraWorkspaceSizeInBytes::Csize_t)::custatevecStatus_t
end

@checked function custatevecSamplerGetSquaredNorm(handle, sampler, norm)
    initialize_context()
    @ccall libcustatevec.custatevecSamplerGetSquaredNorm(handle::custatevecHandle_t,
                                                         sampler::custatevecSamplerDescriptor_t,
                                                         norm::Ptr{Cdouble})::custatevecStatus_t
end

@checked function custatevecSamplerApplySubSVOffset(handle, sampler, subSVOrd, nSubSVs,
                                                    offset, norm)
    initialize_context()
    @ccall libcustatevec.custatevecSamplerApplySubSVOffset(handle::custatevecHandle_t,
                                                           sampler::custatevecSamplerDescriptor_t,
                                                           subSVOrd::Int32, nSubSVs::UInt32,
                                                           offset::Cdouble,
                                                           norm::Cdouble)::custatevecStatus_t
end

@checked function custatevecSamplerSample(handle, sampler, bitStrings, bitOrdering,
                                          bitStringLen, randnums, nShots, output)
    initialize_context()
    @ccall libcustatevec.custatevecSamplerSample(handle::custatevecHandle_t,
                                                 sampler::custatevecSamplerDescriptor_t,
                                                 bitStrings::Ptr{custatevecIndex_t},
                                                 bitOrdering::Ptr{Int32},
                                                 bitStringLen::UInt32,
                                                 randnums::Ptr{Cdouble}, nShots::UInt32,
                                                 output::custatevecSamplerOutput_t)::custatevecStatus_t
end

@checked function custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(handle,
                                                                              svDataType,
                                                                              nIndexBits,
                                                                              permutation,
                                                                              diagonals,
                                                                              diagonalsDataType,
                                                                              targets,
                                                                              nTargets,
                                                                              nControls,
                                                                              extraWorkspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(handle::custatevecHandle_t,
                                                                                     svDataType::cudaDataType_t,
                                                                                     nIndexBits::UInt32,
                                                                                     permutation::PtrOrCuPtr{custatevecIndex_t},
                                                                                     diagonals::CuPtr{Cvoid},
                                                                                     diagonalsDataType::cudaDataType_t,
                                                                                     targets::Ptr{Int32},
                                                                                     nTargets::UInt32,
                                                                                     nControls::UInt32,
                                                                                     extraWorkspaceSizeInBytes::Ptr{Csize_t})::custatevecStatus_t
end

@checked function custatevecApplyGeneralizedPermutationMatrix(handle, sv, svDataType,
                                                              nIndexBits, permutation,
                                                              diagonals, diagonalsDataType,
                                                              adjoint, targets, nTargets,
                                                              controls, controlBitValues,
                                                              nControls, extraWorkspace,
                                                              extraWorkspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecApplyGeneralizedPermutationMatrix(handle::custatevecHandle_t,
                                                                     sv::CuPtr{Cvoid},
                                                                     svDataType::cudaDataType_t,
                                                                     nIndexBits::UInt32,
                                                                     permutation::PtrOrCuPtr{custatevecIndex_t},
                                                                     diagonals::PtrOrCuPtr{Cvoid},
                                                                     diagonalsDataType::cudaDataType_t,
                                                                     adjoint::Int32,
                                                                     targets::Ptr{Int32},
                                                                     nTargets::UInt32,
                                                                     controls::Ptr{Int32},
                                                                     controlBitValues::Ptr{Int32},
                                                                     nControls::UInt32,
                                                                     extraWorkspace::PtrOrCuPtr{Cvoid},
                                                                     extraWorkspaceSizeInBytes::Csize_t)::custatevecStatus_t
end

@checked function custatevecComputeExpectationsOnPauliBasis(handle, sv, svDataType,
                                                            nIndexBits, expectationValues,
                                                            pauliOperatorsArray,
                                                            nPauliOperatorArrays,
                                                            basisBitsArray, nBasisBitsArray)
    initialize_context()
    @ccall libcustatevec.custatevecComputeExpectationsOnPauliBasis(handle::custatevecHandle_t,
                                                                   sv::CuPtr{Cvoid},
                                                                   svDataType::cudaDataType_t,
                                                                   nIndexBits::UInt32,
                                                                   expectationValues::Ptr{Cdouble},
                                                                   pauliOperatorsArray::Ptr{Ptr{custatevecPauli_t}},
                                                                   nPauliOperatorArrays::UInt32,
                                                                   basisBitsArray::Ptr{Ptr{Int32}},
                                                                   nBasisBitsArray::Ptr{UInt32})::custatevecStatus_t
end

@checked function custatevecAccessorCreate(handle, sv, svDataType, nIndexBits, accessor,
                                           bitOrdering, bitOrderingLen, maskBitString,
                                           maskOrdering, maskLen, extraWorkspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecAccessorCreate(handle::custatevecHandle_t,
                                                  sv::CuPtr{Cvoid},
                                                  svDataType::cudaDataType_t,
                                                  nIndexBits::UInt32,
                                                  accessor::Ptr{custatevecAccessorDescriptor_t},
                                                  bitOrdering::Ptr{Int32},
                                                  bitOrderingLen::UInt32,
                                                  maskBitString::Ptr{Int32},
                                                  maskOrdering::Ptr{Int32}, maskLen::UInt32,
                                                  extraWorkspaceSizeInBytes::Ptr{Csize_t})::custatevecStatus_t
end

@checked function custatevecAccessorCreateView(handle, sv, svDataType, nIndexBits, accessor,
                                               bitOrdering, bitOrderingLen, maskBitString,
                                               maskOrdering, maskLen,
                                               extraWorkspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecAccessorCreateView(handle::custatevecHandle_t,
                                                      sv::CuPtr{Cvoid},
                                                      svDataType::cudaDataType_t,
                                                      nIndexBits::UInt32,
                                                      accessor::Ptr{custatevecAccessorDescriptor_t},
                                                      bitOrdering::Ptr{Int32},
                                                      bitOrderingLen::UInt32,
                                                      maskBitString::Ptr{Int32},
                                                      maskOrdering::Ptr{Int32},
                                                      maskLen::UInt32,
                                                      extraWorkspaceSizeInBytes::Ptr{Csize_t})::custatevecStatus_t
end

@checked function custatevecAccessorDestroy(accessor)
    initialize_context()
    @ccall libcustatevec.custatevecAccessorDestroy(accessor::custatevecAccessorDescriptor_t)::custatevecStatus_t
end

@checked function custatevecAccessorSetExtraWorkspace(handle, accessor, extraWorkspace,
                                                      extraWorkspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecAccessorSetExtraWorkspace(handle::custatevecHandle_t,
                                                             accessor::custatevecAccessorDescriptor_t,
                                                             extraWorkspace::Ptr{Cvoid},
                                                             extraWorkspaceSizeInBytes::Csize_t)::custatevecStatus_t
end

@checked function custatevecAccessorGet(handle, accessor, externalBuffer, _begin, _end)
    initialize_context()
    @ccall libcustatevec.custatevecAccessorGet(handle::custatevecHandle_t,
                                               accessor::custatevecAccessorDescriptor_t,
                                               externalBuffer::Ptr{Cvoid},
                                               _begin::custatevecIndex_t,
                                               _end::custatevecIndex_t)::custatevecStatus_t
end

@checked function custatevecAccessorSet(handle, accessor, externalBuffer, _begin, _end)
    initialize_context()
    @ccall libcustatevec.custatevecAccessorSet(handle::custatevecHandle_t,
                                               accessor::custatevecAccessorDescriptor_t,
                                               externalBuffer::Ptr{Cvoid},
                                               _begin::custatevecIndex_t,
                                               _end::custatevecIndex_t)::custatevecStatus_t
end

@checked function custatevecSwapIndexBits(handle, sv, svDataType, nIndexBits, bitSwaps,
                                          nBitSwaps, maskBitString, maskOrdering, maskLen)
    initialize_context()
    @ccall libcustatevec.custatevecSwapIndexBits(handle::custatevecHandle_t,
                                                 sv::CuPtr{Cvoid},
                                                 svDataType::cudaDataType_t,
                                                 nIndexBits::UInt32, bitSwaps::Ptr{int2},
                                                 nBitSwaps::UInt32,
                                                 maskBitString::Ptr{Int32},
                                                 maskOrdering::Ptr{Int32},
                                                 maskLen::UInt32)::custatevecStatus_t
end

@checked function custatevecTestMatrixTypeGetWorkspaceSize(handle, matrixType, matrix,
                                                           matrixDataType, layout, nTargets,
                                                           adjoint, computeType,
                                                           extraWorkspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecTestMatrixTypeGetWorkspaceSize(handle::custatevecHandle_t,
                                                                  matrixType::custatevecMatrixType_t,
                                                                  matrix::PtrOrCuPtr{Cvoid},
                                                                  matrixDataType::cudaDataType_t,
                                                                  layout::custatevecMatrixLayout_t,
                                                                  nTargets::UInt32,
                                                                  adjoint::Int32,
                                                                  computeType::custatevecComputeType_t,
                                                                  extraWorkspaceSizeInBytes::Ptr{Csize_t})::custatevecStatus_t
end

@checked function custatevecTestMatrixType(handle, residualNorm, matrixType, matrix,
                                           matrixDataType, layout, nTargets, adjoint,
                                           computeType, extraWorkspace,
                                           extraWorkspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecTestMatrixType(handle::custatevecHandle_t,
                                                  residualNorm::Ptr{Cdouble},
                                                  matrixType::custatevecMatrixType_t,
                                                  matrix::PtrOrCuPtr{Cvoid},
                                                  matrixDataType::cudaDataType_t,
                                                  layout::custatevecMatrixLayout_t,
                                                  nTargets::UInt32, adjoint::Int32,
                                                  computeType::custatevecComputeType_t,
                                                  extraWorkspace::PtrOrCuPtr{Cvoid},
                                                  extraWorkspaceSizeInBytes::Csize_t)::custatevecStatus_t
end

@checked function custatevecInitializeStateVector(handle, sv, svDataType, nIndexBits,
                                                  svType)
    initialize_context()
    @ccall libcustatevec.custatevecInitializeStateVector(handle::custatevecHandle_t,
                                                         sv::Ptr{Cvoid},
                                                         svDataType::cudaDataType_t,
                                                         nIndexBits::UInt32,
                                                         svType::custatevecStateVectorType_t)::custatevecStatus_t
end

@checked function custatevecMultiDeviceSwapIndexBits(handles, nHandles, subSVs, svDataType,
                                                     nGlobalIndexBits, nLocalIndexBits,
                                                     indexBitSwaps, nIndexBitSwaps,
                                                     maskBitString, maskOrdering, maskLen,
                                                     deviceNetworkType)
    initialize_context()
    @ccall libcustatevec.custatevecMultiDeviceSwapIndexBits(handles::Ptr{custatevecHandle_t},
                                                            nHandles::UInt32,
                                                            subSVs::Ptr{Ptr{Cvoid}},
                                                            svDataType::cudaDataType_t,
                                                            nGlobalIndexBits::UInt32,
                                                            nLocalIndexBits::UInt32,
                                                            indexBitSwaps::Ptr{int2},
                                                            nIndexBitSwaps::UInt32,
                                                            maskBitString::Ptr{Int32},
                                                            maskOrdering::Ptr{Int32},
                                                            maskLen::UInt32,
                                                            deviceNetworkType::custatevecDeviceNetworkType_t)::custatevecStatus_t
end

mutable struct custatevecCommunicator_t end

const custatevecCommunicatorDescriptor_t = Ptr{custatevecCommunicator_t}

@cenum custatevecCommunicatorType_t::UInt32 begin
    CUSTATEVEC_COMMUNICATOR_TYPE_EXTERNAL = 0
    CUSTATEVEC_COMMUNICATOR_TYPE_OPENMPI = 1
    CUSTATEVEC_COMMUNICATOR_TYPE_MPICH = 2
end

@cenum custatevecDataTransferType_t::UInt32 begin
    CUSTATEVEC_DATA_TRANSFER_TYPE_NONE = 0
    CUSTATEVEC_DATA_TRANSFER_TYPE_SEND = 1
    CUSTATEVEC_DATA_TRANSFER_TYPE_RECV = 2
    CUSTATEVEC_DATA_TRANSFER_TYPE_SEND_RECV = 3
end

struct custatevecSVSwapParameters_t
    swapBatchIndex::Int32
    orgSubSVIndex::Int32
    dstSubSVIndex::Int32
    orgSegmentMaskString::NTuple{48,Int32}
    dstSegmentMaskString::NTuple{48,Int32}
    segmentMaskOrdering::NTuple{48,Int32}
    segmentMaskLen::UInt32
    nSegmentBits::UInt32
    dataTransferType::custatevecDataTransferType_t
    transferSize::custatevecIndex_t
end

mutable struct custatevecDistIndexBitSwapScheduler_t end

const custatevecDistIndexBitSwapSchedulerDescriptor_t = Ptr{custatevecDistIndexBitSwapScheduler_t}

mutable struct custatevecSVSwapWorker_t end

const custatevecSVSwapWorkerDescriptor_t = Ptr{custatevecSVSwapWorker_t}

@checked function custatevecCommunicatorCreate(handle, communicator, communicatorType,
                                               soname)
    initialize_context()
    @ccall libcustatevec.custatevecCommunicatorCreate(handle::custatevecHandle_t,
                                                      communicator::Ptr{custatevecCommunicatorDescriptor_t},
                                                      communicatorType::custatevecCommunicatorType_t,
                                                      soname::Cstring)::custatevecStatus_t
end

@checked function custatevecCommunicatorDestroy(handle, communicator)
    initialize_context()
    @ccall libcustatevec.custatevecCommunicatorDestroy(handle::custatevecHandle_t,
                                                       communicator::custatevecCommunicatorDescriptor_t)::custatevecStatus_t
end

@checked function custatevecDistIndexBitSwapSchedulerCreate(handle, scheduler,
                                                            nGlobalIndexBits,
                                                            nLocalIndexBits)
    initialize_context()
    @ccall libcustatevec.custatevecDistIndexBitSwapSchedulerCreate(handle::custatevecHandle_t,
                                                                   scheduler::Ptr{custatevecDistIndexBitSwapSchedulerDescriptor_t},
                                                                   nGlobalIndexBits::UInt32,
                                                                   nLocalIndexBits::UInt32)::custatevecStatus_t
end

@checked function custatevecDistIndexBitSwapSchedulerDestroy(handle, scheduler)
    initialize_context()
    @ccall libcustatevec.custatevecDistIndexBitSwapSchedulerDestroy(handle::custatevecHandle_t,
                                                                    scheduler::custatevecDistIndexBitSwapSchedulerDescriptor_t)::custatevecStatus_t
end

@checked function custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps(handle, scheduler,
                                                                      indexBitSwaps,
                                                                      nIndexBitSwaps,
                                                                      maskBitString,
                                                                      maskOrdering, maskLen,
                                                                      nSwapBatches)
    initialize_context()
    @ccall libcustatevec.custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps(handle::custatevecHandle_t,
                                                                             scheduler::custatevecDistIndexBitSwapSchedulerDescriptor_t,
                                                                             indexBitSwaps::Ptr{int2},
                                                                             nIndexBitSwaps::UInt32,
                                                                             maskBitString::Ptr{Int32},
                                                                             maskOrdering::Ptr{Int32},
                                                                             maskLen::UInt32,
                                                                             nSwapBatches::Ptr{UInt32})::custatevecStatus_t
end

@checked function custatevecDistIndexBitSwapSchedulerGetParameters(handle, scheduler,
                                                                   swapBatchIndex,
                                                                   orgSubSVIndex,
                                                                   parameters)
    initialize_context()
    @ccall libcustatevec.custatevecDistIndexBitSwapSchedulerGetParameters(handle::custatevecHandle_t,
                                                                          scheduler::custatevecDistIndexBitSwapSchedulerDescriptor_t,
                                                                          swapBatchIndex::Int32,
                                                                          orgSubSVIndex::Int32,
                                                                          parameters::Ptr{custatevecSVSwapParameters_t})::custatevecStatus_t
end

@checked function custatevecSVSwapWorkerCreate(handle, svSwapWorker, communicator, orgSubSV,
                                               orgSubSVIndex, orgEvent, svDataType, stream,
                                               extraWorkspaceSizeInBytes,
                                               minTransferWorkspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecSVSwapWorkerCreate(handle::custatevecHandle_t,
                                                      svSwapWorker::Ptr{custatevecSVSwapWorkerDescriptor_t},
                                                      communicator::custatevecCommunicatorDescriptor_t,
                                                      orgSubSV::Ptr{Cvoid},
                                                      orgSubSVIndex::Int32,
                                                      orgEvent::cudaEvent_t,
                                                      svDataType::cudaDataType_t,
                                                      stream::cudaStream_t,
                                                      extraWorkspaceSizeInBytes::Ptr{Csize_t},
                                                      minTransferWorkspaceSizeInBytes::Ptr{Csize_t})::custatevecStatus_t
end

@checked function custatevecSVSwapWorkerDestroy(handle, svSwapWorker)
    initialize_context()
    @ccall libcustatevec.custatevecSVSwapWorkerDestroy(handle::custatevecHandle_t,
                                                       svSwapWorker::custatevecSVSwapWorkerDescriptor_t)::custatevecStatus_t
end

@checked function custatevecSVSwapWorkerSetExtraWorkspace(handle, svSwapWorker,
                                                          extraWorkspace,
                                                          extraWorkspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecSVSwapWorkerSetExtraWorkspace(handle::custatevecHandle_t,
                                                                 svSwapWorker::custatevecSVSwapWorkerDescriptor_t,
                                                                 extraWorkspace::Ptr{Cvoid},
                                                                 extraWorkspaceSizeInBytes::Csize_t)::custatevecStatus_t
end

@checked function custatevecSVSwapWorkerSetTransferWorkspace(handle, svSwapWorker,
                                                             transferWorkspace,
                                                             transferWorkspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecSVSwapWorkerSetTransferWorkspace(handle::custatevecHandle_t,
                                                                    svSwapWorker::custatevecSVSwapWorkerDescriptor_t,
                                                                    transferWorkspace::Ptr{Cvoid},
                                                                    transferWorkspaceSizeInBytes::Csize_t)::custatevecStatus_t
end

@checked function custatevecSVSwapWorkerSetSubSVsP2P(handle, svSwapWorker, dstSubSVsP2P,
                                                     dstSubSVIndicesP2P, dstEvents,
                                                     nDstSubSVsP2P)
    initialize_context()
    @ccall libcustatevec.custatevecSVSwapWorkerSetSubSVsP2P(handle::custatevecHandle_t,
                                                            svSwapWorker::custatevecSVSwapWorkerDescriptor_t,
                                                            dstSubSVsP2P::Ptr{Ptr{Cvoid}},
                                                            dstSubSVIndicesP2P::Ptr{Int32},
                                                            dstEvents::Ptr{cudaEvent_t},
                                                            nDstSubSVsP2P::UInt32)::custatevecStatus_t
end

@checked function custatevecSVSwapWorkerSetParameters(handle, svSwapWorker, parameters,
                                                      peer)
    initialize_context()
    @ccall libcustatevec.custatevecSVSwapWorkerSetParameters(handle::custatevecHandle_t,
                                                             svSwapWorker::custatevecSVSwapWorkerDescriptor_t,
                                                             parameters::Ptr{custatevecSVSwapParameters_t},
                                                             peer::Cint)::custatevecStatus_t
end

@checked function custatevecSVSwapWorkerExecute(handle, svSwapWorker, _begin, _end)
    initialize_context()
    @ccall libcustatevec.custatevecSVSwapWorkerExecute(handle::custatevecHandle_t,
                                                       svSwapWorker::custatevecSVSwapWorkerDescriptor_t,
                                                       _begin::custatevecIndex_t,
                                                       _end::custatevecIndex_t)::custatevecStatus_t
end

@checked function custatevecApplyMatrixBatchedGetWorkspaceSize(handle, svDataType,
                                                               nIndexBits, nSVs, svStride,
                                                               mapType, matrixIndices,
                                                               matrices, matrixDataType,
                                                               layout, adjoint, nMatrices,
                                                               nTargets, nControls,
                                                               computeType,
                                                               extraWorkspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecApplyMatrixBatchedGetWorkspaceSize(handle::custatevecHandle_t,
                                                                      svDataType::cudaDataType_t,
                                                                      nIndexBits::UInt32,
                                                                      nSVs::UInt32,
                                                                      svStride::custatevecIndex_t,
                                                                      mapType::custatevecMatrixMapType_t,
                                                                      matrixIndices::Ptr{Int32},
                                                                      matrices::PtrOrCuPtr{Cvoid},
                                                                      matrixDataType::cudaDataType_t,
                                                                      layout::custatevecMatrixLayout_t,
                                                                      adjoint::Int32,
                                                                      nMatrices::UInt32,
                                                                      nTargets::UInt32,
                                                                      nControls::UInt32,
                                                                      computeType::custatevecComputeType_t,
                                                                      extraWorkspaceSizeInBytes::Ptr{Csize_t})::custatevecStatus_t
end

@checked function custatevecApplyMatrixBatched(handle, batchedSv, svDataType, nIndexBits,
                                               nSVs, svStride, mapType, matrixIndices,
                                               matrices, matrixDataType, layout, adjoint,
                                               nMatrices, targets, nTargets, controls,
                                               controlBitValues, nControls, computeType,
                                               extraWorkspace, extraWorkspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecApplyMatrixBatched(handle::custatevecHandle_t,
                                                      batchedSv::CuPtr{Cvoid},
                                                      svDataType::cudaDataType_t,
                                                      nIndexBits::UInt32, nSVs::UInt32,
                                                      svStride::custatevecIndex_t,
                                                      mapType::custatevecMatrixMapType_t,
                                                      matrixIndices::Ptr{Int32},
                                                      matrices::PtrOrCuPtr{Cvoid},
                                                      matrixDataType::cudaDataType_t,
                                                      layout::custatevecMatrixLayout_t,
                                                      adjoint::Int32, nMatrices::UInt32,
                                                      targets::Ptr{Int32}, nTargets::UInt32,
                                                      controls::Ptr{Int32},
                                                      controlBitValues::Ptr{Int32},
                                                      nControls::UInt32,
                                                      computeType::custatevecComputeType_t,
                                                      extraWorkspace::CuPtr{Cvoid},
                                                      extraWorkspaceSizeInBytes::Csize_t)::custatevecStatus_t
end

@checked function custatevecAbs2SumArrayBatched(handle, batchedSv, svDataType, nIndexBits,
                                                nSVs, svStride, abs2sumArrays,
                                                abs2sumArrayStride, bitOrdering,
                                                bitOrderingLen, maskBitStrings,
                                                maskOrdering, maskLen)
    initialize_context()
    @ccall libcustatevec.custatevecAbs2SumArrayBatched(handle::custatevecHandle_t,
                                                       batchedSv::Ptr{Cvoid},
                                                       svDataType::cudaDataType_t,
                                                       nIndexBits::UInt32, nSVs::UInt32,
                                                       svStride::custatevecIndex_t,
                                                       abs2sumArrays::Ptr{Cdouble},
                                                       abs2sumArrayStride::custatevecIndex_t,
                                                       bitOrdering::Ptr{Int32},
                                                       bitOrderingLen::UInt32,
                                                       maskBitStrings::Ptr{custatevecIndex_t},
                                                       maskOrdering::Ptr{Int32},
                                                       maskLen::UInt32)::custatevecStatus_t
end

@checked function custatevecCollapseByBitStringBatchedGetWorkspaceSize(handle, nSVs,
                                                                       bitStrings, norms,
                                                                       extraWorkspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecCollapseByBitStringBatchedGetWorkspaceSize(handle::custatevecHandle_t,
                                                                              nSVs::UInt32,
                                                                              bitStrings::Ptr{custatevecIndex_t},
                                                                              norms::Ptr{Cdouble},
                                                                              extraWorkspaceSizeInBytes::Ptr{Csize_t})::custatevecStatus_t
end

@checked function custatevecCollapseByBitStringBatched(handle, batchedSv, svDataType,
                                                       nIndexBits, nSVs, svStride,
                                                       bitStrings, bitOrdering,
                                                       bitStringLen, norms, extraWorkspace,
                                                       extraWorkspaceSizeInBytes)
    initialize_context()
    @ccall libcustatevec.custatevecCollapseByBitStringBatched(handle::custatevecHandle_t,
                                                              batchedSv::Ptr{Cvoid},
                                                              svDataType::cudaDataType_t,
                                                              nIndexBits::UInt32,
                                                              nSVs::UInt32,
                                                              svStride::custatevecIndex_t,
                                                              bitStrings::Ptr{custatevecIndex_t},
                                                              bitOrdering::Ptr{Int32},
                                                              bitStringLen::UInt32,
                                                              norms::Ptr{Cdouble},
                                                              extraWorkspace::Ptr{Cvoid},
                                                              extraWorkspaceSizeInBytes::Csize_t)::custatevecStatus_t
end

@checked function custatevecMeasureBatched(handle, batchedSv, svDataType, nIndexBits, nSVs,
                                           svStride, bitStrings, bitOrdering, bitStringLen,
                                           randnums, collapse)
    initialize_context()
    @ccall libcustatevec.custatevecMeasureBatched(handle::custatevecHandle_t,
                                                  batchedSv::Ptr{Cvoid},
                                                  svDataType::cudaDataType_t,
                                                  nIndexBits::UInt32, nSVs::UInt32,
                                                  svStride::custatevecIndex_t,
                                                  bitStrings::Ptr{custatevecIndex_t},
                                                  bitOrdering::Ptr{Int32},
                                                  bitStringLen::UInt32,
                                                  randnums::Ptr{Cdouble},
                                                  collapse::custatevecCollapseOp_t)::custatevecStatus_t
end

mutable struct custatevecSubSVMigratorDescriptor end

const custatevecSubSVMigratorDescriptor_t = Ptr{custatevecSubSVMigratorDescriptor}

@checked function custatevecSubSVMigratorCreate(handle, migrator, deviceSlots, svDataType,
                                                nDeviceSlots, nLocalIndexBits)
    initialize_context()
    @ccall libcustatevec.custatevecSubSVMigratorCreate(handle::custatevecHandle_t,
                                                       migrator::Ptr{custatevecSubSVMigratorDescriptor_t},
                                                       deviceSlots::Ptr{Cvoid},
                                                       svDataType::cudaDataType_t,
                                                       nDeviceSlots::Cint,
                                                       nLocalIndexBits::Cint)::custatevecStatus_t
end

@checked function custatevecSubSVMigratorDestroy(handle, migrator)
    initialize_context()
    @ccall libcustatevec.custatevecSubSVMigratorDestroy(handle::custatevecHandle_t,
                                                        migrator::custatevecSubSVMigratorDescriptor_t)::custatevecStatus_t
end

@checked function custatevecSubSVMigratorMigrate(handle, migrator, deviceSlotIndex,
                                                 srcSubSV, dstSubSV, _begin, _end)
    initialize_context()
    @ccall libcustatevec.custatevecSubSVMigratorMigrate(handle::custatevecHandle_t,
                                                        migrator::custatevecSubSVMigratorDescriptor_t,
                                                        deviceSlotIndex::Cint,
                                                        srcSubSV::Ptr{Cvoid},
                                                        dstSubSV::Ptr{Cvoid},
                                                        _begin::custatevecIndex_t,
                                                        _end::custatevecIndex_t)::custatevecStatus_t
end

const CUSTATEVEC_ALLOCATOR_NAME_LEN = 64

const CUSTATEVEC_MAX_SEGMENT_MASK_SIZE = 48
