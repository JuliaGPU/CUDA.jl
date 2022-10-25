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

macro check(ex, errs...)
    check = :(isequal(err, CUSTATEVEC_STATUS_ALLOC_FAILED))
    for err in errs
        check = :($check || isequal(err, $(esc(err))))
    end

    quote
        res = @retry_reclaim err -> $check $(esc(ex))
        if res != CUSTATEVEC_STATUS_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
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
    CUSTATEVEC_STATUS_MAX_VALUE = 11
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

@cenum custatevecDeviceNetworkType_t::UInt32 begin
    CUSTATEVEC_DEVICE_NETWORK_TYPE_SWITCH = 1
    CUSTATEVEC_DEVICE_NETWORK_TYPE_FULLMESH = 2
end

@checked function custatevecCreate(handle)
    initialize_context()
    ccall((:custatevecCreate, libcustatevec), custatevecStatus_t,
          (Ptr{custatevecHandle_t},), handle)
end

@checked function custatevecDestroy(handle)
    initialize_context()
    ccall((:custatevecDestroy, libcustatevec), custatevecStatus_t, (custatevecHandle_t,),
          handle)
end

@checked function custatevecGetDefaultWorkspaceSize(handle, workspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecGetDefaultWorkspaceSize, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, Ptr{Csize_t}), handle, workspaceSizeInBytes)
end

@checked function custatevecSetWorkspace(handle, workspace, workspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecSetWorkspace, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, Ptr{Cvoid}, Csize_t), handle, workspace,
          workspaceSizeInBytes)
end

function custatevecGetErrorName(status)
    initialize_context()
    ccall((:custatevecGetErrorName, libcustatevec), Cstring, (custatevecStatus_t,), status)
end

function custatevecGetErrorString(status)
    ccall((:custatevecGetErrorString, libcustatevec), Cstring, (custatevecStatus_t,),
          status)
end

@checked function custatevecGetProperty(type, value)
    initialize_context()
    ccall((:custatevecGetProperty, libcustatevec), custatevecStatus_t,
          (libraryPropertyType, Ptr{Int32}), type, value)
end

# no prototype is found for this function at custatevec.h:519:8, please use with caution
function custatevecGetVersion()
    ccall((:custatevecGetVersion, libcustatevec), Csize_t, ())
end

@checked function custatevecSetStream(handle, streamId)
    initialize_context()
    ccall((:custatevecSetStream, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, cudaStream_t), handle, streamId)
end

@checked function custatevecGetStream(handle, streamId)
    initialize_context()
    ccall((:custatevecGetStream, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, Ptr{cudaStream_t}), handle, streamId)
end

@checked function custatevecLoggerSetCallback(callback)
    ccall((:custatevecLoggerSetCallback, libcustatevec), custatevecStatus_t,
          (custatevecLoggerCallback_t,), callback)
end

@checked function custatevecLoggerSetCallbackData(callback, userData)
    initialize_context()
    ccall((:custatevecLoggerSetCallbackData, libcustatevec), custatevecStatus_t,
          (custatevecLoggerCallbackData_t, Ptr{Cvoid}), callback, userData)
end

@checked function custatevecLoggerSetFile(file)
    ccall((:custatevecLoggerSetFile, libcustatevec), custatevecStatus_t, (Ptr{Libc.FILE},),
          file)
end

@checked function custatevecLoggerOpenFile(logFile)
    ccall((:custatevecLoggerOpenFile, libcustatevec), custatevecStatus_t, (Cstring,),
          logFile)
end

@checked function custatevecLoggerSetLevel(level)
    initialize_context()
    ccall((:custatevecLoggerSetLevel, libcustatevec), custatevecStatus_t, (Int32,), level)
end

@checked function custatevecLoggerSetMask(mask)
    ccall((:custatevecLoggerSetMask, libcustatevec), custatevecStatus_t, (Int32,), mask)
end

# no prototype is found for this function at custatevec.h:619:1, please use with caution
@checked function custatevecLoggerForceDisable()
    ccall((:custatevecLoggerForceDisable, libcustatevec), custatevecStatus_t, ())
end

@checked function custatevecGetDeviceMemHandler(handle, handler)
    initialize_context()
    ccall((:custatevecGetDeviceMemHandler, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, Ptr{custatevecDeviceMemHandler_t}), handle, handler)
end

@checked function custatevecSetDeviceMemHandler(handle, handler)
    initialize_context()
    ccall((:custatevecSetDeviceMemHandler, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, Ptr{custatevecDeviceMemHandler_t}), handle, handler)
end

@checked function custatevecAbs2SumOnZBasis(handle, sv, svDataType, nIndexBits, abs2sum0,
                                            abs2sum1, basisBits, nBasisBits)
    initialize_context()
    ccall((:custatevecAbs2SumOnZBasis, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Cdouble},
           Ptr{Cdouble}, Ptr{Int32}, UInt32), handle, sv, svDataType, nIndexBits, abs2sum0,
          abs2sum1, basisBits, nBasisBits)
end

@checked function custatevecAbs2SumArray(handle, sv, svDataType, nIndexBits, abs2sum,
                                         bitOrdering, bitOrderingLen, maskBitString,
                                         maskOrdering, maskLen)
    initialize_context()
    ccall((:custatevecAbs2SumArray, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Cdouble}, Ptr{Int32},
           UInt32, Ptr{Int32}, Ptr{Int32}, UInt32), handle, sv, svDataType, nIndexBits,
          abs2sum, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen)
end

@checked function custatevecCollapseOnZBasis(handle, sv, svDataType, nIndexBits, parity,
                                             basisBits, nBasisBits, norm)
    initialize_context()
    ccall((:custatevecCollapseOnZBasis, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Int32, Ptr{Int32},
           UInt32, Cdouble), handle, sv, svDataType, nIndexBits, parity, basisBits,
          nBasisBits, norm)
end

@checked function custatevecCollapseByBitString(handle, sv, svDataType, nIndexBits,
                                                bitString, bitOrdering, bitStringLen, norm)
    initialize_context()
    ccall((:custatevecCollapseByBitString, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Int32}, Ptr{Int32},
           UInt32, Cdouble), handle, sv, svDataType, nIndexBits, bitString, bitOrdering,
          bitStringLen, norm)
end

@checked function custatevecMeasureOnZBasis(handle, sv, svDataType, nIndexBits, parity,
                                            basisBits, nBasisBits, randnum, collapse)
    initialize_context()
    ccall((:custatevecMeasureOnZBasis, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Int32}, Ptr{Int32},
           UInt32, Cdouble, custatevecCollapseOp_t), handle, sv, svDataType, nIndexBits,
          parity, basisBits, nBasisBits, randnum, collapse)
end

@checked function custatevecBatchMeasure(handle, sv, svDataType, nIndexBits, bitString,
                                         bitOrdering, bitStringLen, randnum, collapse)
    initialize_context()
    ccall((:custatevecBatchMeasure, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Int32}, Ptr{Int32},
           UInt32, Cdouble, custatevecCollapseOp_t), handle, sv, svDataType, nIndexBits,
          bitString, bitOrdering, bitStringLen, randnum, collapse)
end

@checked function custatevecBatchMeasureWithOffset(handle, sv, svDataType, nIndexBits,
                                                   bitString, bitOrdering, bitStringLen,
                                                   randnum, collapse, offset, abs2sum)
    initialize_context()
    ccall((:custatevecBatchMeasureWithOffset, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Int32}, Ptr{Int32},
           UInt32, Cdouble, custatevecCollapseOp_t, Cdouble, Cdouble), handle, sv,
          svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, randnum, collapse,
          offset, abs2sum)
end

@checked function custatevecApplyPauliRotation(handle, sv, svDataType, nIndexBits, theta,
                                               paulis, targets, nTargets, controls,
                                               controlBitValues, nControls)
    initialize_context()
    ccall((:custatevecApplyPauliRotation, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Cdouble,
           Ptr{custatevecPauli_t}, Ptr{Int32}, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32),
          handle, sv, svDataType, nIndexBits, theta, paulis, targets, nTargets, controls,
          controlBitValues, nControls)
end

@checked function custatevecApplyMatrixGetWorkspaceSize(handle, svDataType, nIndexBits,
                                                        matrix, matrixDataType, layout,
                                                        adjoint, nTargets, nControls,
                                                        computeType,
                                                        extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecApplyMatrixGetWorkspaceSize, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, cudaDataType_t, UInt32, Ptr{Cvoid}, cudaDataType_t,
           custatevecMatrixLayout_t, Int32, UInt32, UInt32, custatevecComputeType_t,
           Ptr{Csize_t}), handle, svDataType, nIndexBits, matrix, matrixDataType, layout,
          adjoint, nTargets, nControls, computeType, extraWorkspaceSizeInBytes)
end

@checked function custatevecApplyMatrix(handle, sv, svDataType, nIndexBits, matrix,
                                        matrixDataType, layout, adjoint, targets, nTargets,
                                        controls, controlBitValues, nControls, computeType,
                                        extraWorkspace, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecApplyMatrix, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, CuPtr{Cvoid}, cudaDataType_t, UInt32, PtrOrCuPtr{Cvoid},
           cudaDataType_t, custatevecMatrixLayout_t, Int32, Ptr{Int32}, UInt32, Ptr{Int32},
           Ptr{Int32}, UInt32, custatevecComputeType_t, CuPtr{Cvoid}, Csize_t), handle, sv,
          svDataType, nIndexBits, matrix, matrixDataType, layout, adjoint, targets,
          nTargets, controls, controlBitValues, nControls, computeType, extraWorkspace,
          extraWorkspaceSizeInBytes)
end

@checked function custatevecComputeExpectationGetWorkspaceSize(handle, svDataType,
                                                               nIndexBits, matrix,
                                                               matrixDataType, layout,
                                                               nBasisBits, computeType,
                                                               extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecComputeExpectationGetWorkspaceSize, libcustatevec),
          custatevecStatus_t,
          (custatevecHandle_t, cudaDataType_t, UInt32, PtrOrCuPtr{Cvoid}, cudaDataType_t,
           custatevecMatrixLayout_t, UInt32, custatevecComputeType_t, Ptr{Csize_t}), handle,
          svDataType, nIndexBits, matrix, matrixDataType, layout, nBasisBits, computeType,
          extraWorkspaceSizeInBytes)
end

@checked function custatevecComputeExpectation(handle, sv, svDataType, nIndexBits,
                                               expectationValue, expectationDataType,
                                               residualNorm, matrix, matrixDataType, layout,
                                               basisBits, nBasisBits, computeType,
                                               extraWorkspace, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecComputeExpectation, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, CuPtr{Cvoid}, cudaDataType_t, UInt32, Ptr{Cvoid},
           cudaDataType_t, Ptr{Cdouble}, PtrOrCuPtr{Cvoid}, cudaDataType_t,
           custatevecMatrixLayout_t, Ptr{Int32}, UInt32, custatevecComputeType_t,
           CuPtr{Cvoid}, Csize_t), handle, sv, svDataType, nIndexBits, expectationValue,
          expectationDataType, residualNorm, matrix, matrixDataType, layout, basisBits,
          nBasisBits, computeType, extraWorkspace, extraWorkspaceSizeInBytes)
end

@checked function custatevecSamplerCreate(handle, sv, svDataType, nIndexBits, sampler,
                                          nMaxShots, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecSamplerCreate, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, CuPtr{Cvoid}, cudaDataType_t, UInt32,
           Ptr{custatevecSamplerDescriptor_t}, UInt32, Ptr{Csize_t}), handle, sv,
          svDataType, nIndexBits, sampler, nMaxShots, extraWorkspaceSizeInBytes)
end

@checked function custatevecSamplerDestroy(sampler)
    initialize_context()
    ccall((:custatevecSamplerDestroy, libcustatevec), custatevecStatus_t,
          (custatevecSamplerDescriptor_t,), sampler)
end

@checked function custatevecSamplerPreprocess(handle, sampler, extraWorkspace,
                                              extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecSamplerPreprocess, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, custatevecSamplerDescriptor_t, CuPtr{Cvoid}, Csize_t),
          handle, sampler, extraWorkspace, extraWorkspaceSizeInBytes)
end

@checked function custatevecSamplerGetSquaredNorm(handle, sampler, norm)
    initialize_context()
    ccall((:custatevecSamplerGetSquaredNorm, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, custatevecSamplerDescriptor_t, Ptr{Cdouble}), handle,
          sampler, norm)
end

@checked function custatevecSamplerApplySubSVOffset(handle, sampler, subSVOrd, nSubSVs,
                                                    offset, norm)
    initialize_context()
    ccall((:custatevecSamplerApplySubSVOffset, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, custatevecSamplerDescriptor_t, Int32, UInt32, Cdouble,
           Cdouble), handle, sampler, subSVOrd, nSubSVs, offset, norm)
end

@checked function custatevecSamplerSample(handle, sampler, bitStrings, bitOrdering,
                                          bitStringLen, randnums, nShots, output)
    initialize_context()
    ccall((:custatevecSamplerSample, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, custatevecSamplerDescriptor_t, Ptr{custatevecIndex_t},
           Ptr{Int32}, UInt32, Ptr{Cdouble}, UInt32, custatevecSamplerOutput_t), handle,
          sampler, bitStrings, bitOrdering, bitStringLen, randnums, nShots, output)
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
    ccall((:custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize, libcustatevec),
          custatevecStatus_t,
          (custatevecHandle_t, cudaDataType_t, UInt32, Ptr{custatevecIndex_t}, Ptr{Cvoid},
           cudaDataType_t, Ptr{Int32}, UInt32, UInt32, Ptr{Csize_t}), handle, svDataType,
          nIndexBits, permutation, diagonals, diagonalsDataType, targets, nTargets,
          nControls, extraWorkspaceSizeInBytes)
end

@checked function custatevecApplyGeneralizedPermutationMatrix(handle, sv, svDataType,
                                                              nIndexBits, permutation,
                                                              diagonals, diagonalsDataType,
                                                              adjoint, targets, nTargets,
                                                              controls, controlBitValues,
                                                              nControls, extraWorkspace,
                                                              extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecApplyGeneralizedPermutationMatrix, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{custatevecIndex_t},
           Ptr{Cvoid}, cudaDataType_t, Int32, Ptr{Int32}, UInt32, Ptr{Int32}, Ptr{Int32},
           UInt32, Ptr{Cvoid}, Csize_t), handle, sv, svDataType, nIndexBits, permutation,
          diagonals, diagonalsDataType, adjoint, targets, nTargets, controls,
          controlBitValues, nControls, extraWorkspace, extraWorkspaceSizeInBytes)
end

@checked function custatevecComputeExpectationsOnPauliBasis(handle, sv, svDataType,
                                                            nIndexBits, expectationValues,
                                                            pauliOperatorsArray,
                                                            nPauliOperatorArrays,
                                                            basisBitsArray, nBasisBitsArray)
    initialize_context()
    ccall((:custatevecComputeExpectationsOnPauliBasis, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Cdouble},
           Ptr{Ptr{custatevecPauli_t}}, UInt32, Ptr{Ptr{Int32}}, Ptr{UInt32}), handle, sv,
          svDataType, nIndexBits, expectationValues, pauliOperatorsArray,
          nPauliOperatorArrays, basisBitsArray, nBasisBitsArray)
end

@checked function custatevecAccessorCreate(handle, sv, svDataType, nIndexBits, accessor,
                                           bitOrdering, bitOrderingLen, maskBitString,
                                           maskOrdering, maskLen, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecAccessorCreate, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32,
           Ptr{custatevecAccessorDescriptor_t}, Ptr{Int32}, UInt32, Ptr{Int32}, Ptr{Int32},
           UInt32, Ptr{Csize_t}), handle, sv, svDataType, nIndexBits, accessor, bitOrdering,
          bitOrderingLen, maskBitString, maskOrdering, maskLen, extraWorkspaceSizeInBytes)
end

@checked function custatevecAccessorCreateView(handle, sv, svDataType, nIndexBits, accessor,
                                               bitOrdering, bitOrderingLen, maskBitString,
                                               maskOrdering, maskLen,
                                               extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecAccessorCreateView, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32,
           Ptr{custatevecAccessorDescriptor_t}, Ptr{Int32}, UInt32, Ptr{Int32}, Ptr{Int32},
           UInt32, Ptr{Csize_t}), handle, sv, svDataType, nIndexBits, accessor, bitOrdering,
          bitOrderingLen, maskBitString, maskOrdering, maskLen, extraWorkspaceSizeInBytes)
end

@checked function custatevecAccessorDestroy(accessor)
    initialize_context()
    ccall((:custatevecAccessorDestroy, libcustatevec), custatevecStatus_t,
          (custatevecAccessorDescriptor_t,), accessor)
end

@checked function custatevecAccessorSetExtraWorkspace(handle, accessor, extraWorkspace,
                                                      extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecAccessorSetExtraWorkspace, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, custatevecAccessorDescriptor_t, Ptr{Cvoid}, Csize_t), handle,
          accessor, extraWorkspace, extraWorkspaceSizeInBytes)
end

@checked function custatevecAccessorGet(handle, accessor, externalBuffer, _begin, _end)
    initialize_context()
    ccall((:custatevecAccessorGet, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, custatevecAccessorDescriptor_t, Ptr{Cvoid},
           custatevecIndex_t, custatevecIndex_t), handle, accessor, externalBuffer, _begin,
          _end)
end

@checked function custatevecAccessorSet(handle, accessor, externalBuffer, _begin, _end)
    initialize_context()
    ccall((:custatevecAccessorSet, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, custatevecAccessorDescriptor_t, Ptr{Cvoid},
           custatevecIndex_t, custatevecIndex_t), handle, accessor, externalBuffer, _begin,
          _end)
end

@checked function custatevecSwapIndexBits(handle, sv, svDataType, nIndexBits, bitSwaps,
                                          nBitSwaps, maskBitString, maskOrdering, maskLen)
    initialize_context()
    ccall((:custatevecSwapIndexBits, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{int2}, UInt32,
           Ptr{Int32}, Ptr{Int32}, UInt32), handle, sv, svDataType, nIndexBits, bitSwaps,
          nBitSwaps, maskBitString, maskOrdering, maskLen)
end

@checked function custatevecTestMatrixTypeGetWorkspaceSize(handle, matrixType, matrix,
                                                           matrixDataType, layout, nTargets,
                                                           adjoint, computeType,
                                                           extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecTestMatrixTypeGetWorkspaceSize, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, custatevecMatrixType_t, Ptr{Cvoid}, cudaDataType_t,
           custatevecMatrixLayout_t, UInt32, Int32, custatevecComputeType_t, Ptr{Csize_t}),
          handle, matrixType, matrix, matrixDataType, layout, nTargets, adjoint,
          computeType, extraWorkspaceSizeInBytes)
end

@checked function custatevecTestMatrixType(handle, residualNorm, matrixType, matrix,
                                           matrixDataType, layout, nTargets, adjoint,
                                           computeType, extraWorkspace,
                                           extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecTestMatrixType, libcustatevec), custatevecStatus_t,
          (custatevecHandle_t, Ptr{Cdouble}, custatevecMatrixType_t, Ptr{Cvoid},
           cudaDataType_t, custatevecMatrixLayout_t, UInt32, Int32, custatevecComputeType_t,
           Ptr{Cvoid}, Csize_t), handle, residualNorm, matrixType, matrix, matrixDataType,
          layout, nTargets, adjoint, computeType, extraWorkspace, extraWorkspaceSizeInBytes)
end

@checked function custatevecMultiDeviceSwapIndexBits(handles, nHandles, subSVs, svDataType,
                                                     nGlobalIndexBits, nLocalIndexBits,
                                                     indexBitSwaps, nIndexBitSwaps,
                                                     maskBitString, maskOrdering, maskLen,
                                                     deviceNetworkType)
    initialize_context()
    ccall((:custatevecMultiDeviceSwapIndexBits, libcustatevec), custatevecStatus_t,
          (Ptr{custatevecHandle_t}, UInt32, Ptr{Ptr{Cvoid}}, cudaDataType_t, UInt32, UInt32,
           Ptr{int2}, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32,
           custatevecDeviceNetworkType_t), handles, nHandles, subSVs, svDataType,
          nGlobalIndexBits, nLocalIndexBits, indexBitSwaps, nIndexBitSwaps, maskBitString,
          maskOrdering, maskLen, deviceNetworkType)
end

const CUSTATEVEC_ALLOCATOR_NAME_LEN = 64
