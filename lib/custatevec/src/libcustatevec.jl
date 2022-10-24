function custatevecGetErrorName(status)
    initialize_context()
    ccall((:custatevecGetErrorName, libcustatevec), Cstring, (custatevecStatus_t,), status)
end

function custatevecCreate(handle)
    initialize_context()
    ccall((:custatevecCreate, libcustatevec), custatevecStatus_t, (Ptr{custatevecHandle_t},), handle)
end

function custatevecCollapseOnZBasis(handle, sv, svDataType, nIndexBits, parity, basisBits, nBasisBits, norm)
    initialize_context()
    ccall((:custatevecCollapseOnZBasis, libcustatevec), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Int32, Ptr{Int32}, UInt32, Cdouble), handle, sv, svDataType, nIndexBits, parity, basisBits, nBasisBits, norm)
end

function custatevecBatchMeasure(handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, randnum, collapse)
    initialize_context()
    ccall((:custatevecBatchMeasure, libcustatevec), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32, Cdouble, custatevecCollapseOp_t), handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, randnum, collapse)
end

function custatevecLoggerSetMask(mask)
    ccall((:custatevecLoggerSetMask, libcustatevec), custatevecStatus_t, (Int32,), mask)
end

function custatevecGetStream(handle, streamId)
    initialize_context()
    ccall((:custatevecGetStream, libcustatevec), custatevecStatus_t, (custatevecHandle_t, Ptr{CUstream}), handle, streamId)
end

function custatevecGetVersion()
    ccall((:custatevecGetVersion, libcustatevec), Csize_t, ())
end

function custatevecLoggerForceDisable()
    ccall((:custatevecLoggerForceDisable, libcustatevec), custatevecStatus_t, ())
end

function custatevecGetDefaultWorkspaceSize(handle, workspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecGetDefaultWorkspaceSize, libcustatevec), custatevecStatus_t, (custatevecHandle_t, Ptr{Csize_t}), handle, workspaceSizeInBytes)
end

function custatevecSetStream(handle, streamId)
    initialize_context()
    ccall((:custatevecSetStream, libcustatevec), custatevecStatus_t, (custatevecHandle_t, CUstream), handle, streamId)
end

function custatevecSetWorkspace(handle, workspace, workspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecSetWorkspace, libcustatevec), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, Csize_t), handle, workspace, workspaceSizeInBytes)
end

function custatevecAbs2SumArray(handle, sv, svDataType, nIndexBits, abs2sum, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen)
    initialize_context()
    ccall((:custatevecAbs2SumArray, libcustatevec), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Cdouble}, Ptr{Int32}, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32), handle, sv, svDataType, nIndexBits, abs2sum, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen)
end

function custatevecGetProperty(type, value)
    initialize_context()
    ccall((:custatevecGetProperty, libcustatevec), custatevecStatus_t, (libraryPropertyType, Ptr{Int32}), type, value)
end

function custatevecAbs2SumOnZBasis(handle, sv, svDataType, nIndexBits, abs2sum0, abs2sum1, basisBits, nBasisBits)
    initialize_context()
    ccall((:custatevecAbs2SumOnZBasis, libcustatevec), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int32}, UInt32), handle, sv, svDataType, nIndexBits, abs2sum0, abs2sum1, basisBits, nBasisBits)
end

function custatevecLoggerOpenFile(logFile)
    ccall((:custatevecLoggerOpenFile, libcustatevec), custatevecStatus_t, (Cstring,), logFile)
end

function custatevecDestroy(handle)
    initialize_context()
    ccall((:custatevecDestroy, libcustatevec), custatevecStatus_t, (custatevecHandle_t,), handle)
end

function custatevecLoggerSetLevel(level)
    initialize_context()
    ccall((:custatevecLoggerSetLevel, libcustatevec), custatevecStatus_t, (Int32,), level)
end

function custatevecLoggerSetFile(file)
    ccall((:custatevecLoggerSetFile, libcustatevec), custatevecStatus_t, (Ptr{Cvoid},), file)
end

function custatevecGetErrorString(status)
    ccall((:custatevecGetErrorString, libcustatevec), Cstring, (custatevecStatus_t,), status)
end

function custatevecLoggerSetCallback(callback)
    ccall((:custatevecLoggerSetCallback, libcustatevec), custatevecStatus_t, (custatevecLoggerCallback_t,), callback)
end

function custatevecCollapseByBitString(handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, norm)
    initialize_context()
    ccall((:custatevecCollapseByBitString, libcustatevec), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32, Cdouble), handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, norm)
end

function custatevecMeasureOnZBasis(handle, sv, svDataType, nIndexBits, parity, basisBits, nBasisBits, randnum, collapse)
    initialize_context()
    ccall((:custatevecMeasureOnZBasis, libcustatevec), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32, Cdouble, custatevecCollapseOp_t), handle, sv, svDataType, nIndexBits, parity, basisBits, nBasisBits, randnum, collapse)
end

function custatevecSamplerGetSquaredNorm(handle, sampler, norm)
    initialize_context()
    ccall((:custatevecSamplerGetSquaredNorm, libcustatevec), custatevecStatus_t, (custatevecHandle_t, custatevecSamplerDescriptor_t, Ptr{Cdouble}), handle, sampler, norm)
end

function custatevecMultiDeviceSwapIndexBits(handles, nHandles, subSVs, svDataType, nGlobalIndexBits, nLocalIndexBits, indexBitSwaps, nIndexBitSwaps, maskBitString, maskOrdering, maskLen, deviceNetworkType)
    initialize_context()
    ccall((:custatevecMultiDeviceSwapIndexBits, libcustatevec), custatevecStatus_t, (Ptr{custatevecHandle_t}, UInt32, Ptr{Ptr{Cvoid}}, cudaDataType_t, UInt32, UInt32, Ptr{Tuple{Int32,Int32}}, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32, custatevecDeviceNetworkType_t), handles, nHandles, subSVs, svDataType, nGlobalIndexBits, nLocalIndexBits, indexBitSwaps, nIndexBitSwaps, maskBitString, maskOrdering, maskLen, deviceNetworkType)
end

function custatevecBatchMeasureWithOffset(handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, randnum, collapse, offset, abs2sum)
    initialize_context()
    ccall((:custatevecBatchMeasureWithOffset, libcustatevec), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32, Cdouble, custatevecCollapseOp_t, Cdouble, Cdouble), handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, randnum, collapse, offset, abs2sum)
end

function custatevecComputeExpectationGetWorkspaceSize(handle, svDataType, nIndexBits, matrix, matrixDataType, layout, nBasisBits, computeType, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecComputeExpectationGetWorkspaceSize, libcustatevec), custatevecStatus_t, (custatevecHandle_t, cudaDataType_t, UInt32, PtrOrCuPtr{Cvoid}, cudaDataType_t, custatevecMatrixLayout_t, UInt32, custatevecComputeType_t, Ptr{Csize_t}), handle, svDataType, nIndexBits, matrix, matrixDataType, layout, nBasisBits, computeType, extraWorkspaceSizeInBytes)
end

function custatevecSetDeviceMemHandler(handle, handler)
    initialize_context()
    ccall((:custatevecSetDeviceMemHandler, libcustatevec), custatevecStatus_t, (custatevecHandle_t, Ptr{custatevecDeviceMemHandler_t}), handle, handler)
end

function custatevecAccessorSetExtraWorkspace(handle, accessor, extraWorkspace, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecAccessorSetExtraWorkspace, libcustatevec), custatevecStatus_t, (custatevecHandle_t, custatevecAccessorDescriptor_t, Ptr{Cvoid}, Csize_t), handle, accessor, extraWorkspace, extraWorkspaceSizeInBytes)
end

function custatevecAccessorCreate(handle, sv, svDataType, nIndexBits, accessor, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecAccessorCreate, libcustatevec), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{custatevecAccessorDescriptor_t}, Ptr{Int32}, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32, Ptr{Csize_t}), handle, sv, svDataType, nIndexBits, accessor, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen, extraWorkspaceSizeInBytes)
end

function custatevecAccessorDestroy(accessor)
    initialize_context()
    ccall((:custatevecAccessorDestroy, libcustatevec), custatevecStatus_t, (custatevecAccessorDescriptor_t,), accessor)
end

function custatevecLoggerSetCallbackData(callback, userData)
    initialize_context()
    ccall((:custatevecLoggerSetCallbackData, libcustatevec), custatevecStatus_t, (custatevecLoggerCallbackData_t, Ptr{Cvoid}), callback, userData)
end

function custatevecAccessorSet(handle, accessor, externalBuffer, _begin, _end)
    initialize_context()
    ccall((:custatevecAccessorSet, libcustatevec), custatevecStatus_t, (custatevecHandle_t, custatevecAccessorDescriptor_t, Ptr{Cvoid}, custatevecIndex_t, custatevecIndex_t), handle, accessor, externalBuffer, _begin, _end)
end

function custatevecSamplerSample(handle, sampler, bitStrings, bitOrdering, bitStringLen, randnums, nShots, output)
    initialize_context()
    ccall((:custatevecSamplerSample, libcustatevec), custatevecStatus_t, (custatevecHandle_t, custatevecSamplerDescriptor_t, Ptr{custatevecIndex_t}, Ptr{Int32}, UInt32, Ptr{Cdouble}, UInt32, custatevecSamplerOutput_t), handle, sampler, bitStrings, bitOrdering, bitStringLen, randnums, nShots, output)
end

function custatevecSwapIndexBits(handle, sv, svDataType, nIndexBits, bitSwaps, nBitSwaps, maskBitString, maskOrdering, maskLen)
    initialize_context()
    ccall((:custatevecSwapIndexBits, libcustatevec), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Tuple{Int32,Int32}}, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32), handle, sv, svDataType, nIndexBits, bitSwaps, nBitSwaps, maskBitString, maskOrdering, maskLen)
end

function custatevecSamplerDestroy(sampler)
    initialize_context()
    ccall((:custatevecSamplerDestroy, libcustatevec), custatevecStatus_t, (custatevecSamplerDescriptor_t,), sampler)
end

function custatevecApplyMatrixGetWorkspaceSize(handle, svDataType, nIndexBits, matrix, matrixDataType, layout, adjoint, nTargets, nControls, computeType, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecApplyMatrixGetWorkspaceSize, libcustatevec), custatevecStatus_t, (custatevecHandle_t, cudaDataType_t, UInt32, Ptr{Cvoid}, cudaDataType_t, custatevecMatrixLayout_t, Int32, UInt32, UInt32, custatevecComputeType_t, Ptr{Csize_t}), handle, svDataType, nIndexBits, matrix, matrixDataType, layout, adjoint, nTargets, nControls, computeType, extraWorkspaceSizeInBytes)
end

function custatevecTestMatrixTypeGetWorkspaceSize(handle, matrixType, matrix, matrixDataType, layout, nTargets, adjoint, computeType, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecTestMatrixTypeGetWorkspaceSize, libcustatevec), custatevecStatus_t, (custatevecHandle_t, custatevecMatrixType_t, Ptr{Cvoid}, cudaDataType_t, custatevecMatrixLayout_t, UInt32, Int32, custatevecComputeType_t, Ptr{Csize_t}), handle, matrixType, matrix, matrixDataType, layout, nTargets, adjoint, computeType, extraWorkspaceSizeInBytes)
end

function custatevecTestMatrixType(handle, residualNorm, matrixType, matrix, matrixDataType, layout, nTargets, adjoint, computeType, extraWorkspace, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecTestMatrixType, libcustatevec), custatevecStatus_t, (custatevecHandle_t, Ptr{Cdouble}, custatevecMatrixType_t, Ptr{Cvoid}, cudaDataType_t, custatevecMatrixLayout_t, UInt32, Int32, custatevecComputeType_t, Ptr{Cvoid}, Csize_t), handle, residualNorm, matrixType, matrix, matrixDataType, layout, nTargets, adjoint, computeType, extraWorkspace, extraWorkspaceSizeInBytes)
end

function custatevecAccessorGet(handle, accessor, externalBuffer, _begin, _end)
    initialize_context()
    ccall((:custatevecAccessorGet, libcustatevec), custatevecStatus_t, (custatevecHandle_t, custatevecAccessorDescriptor_t, Ptr{Cvoid}, custatevecIndex_t, custatevecIndex_t), handle, accessor, externalBuffer, _begin, _end)
end

function custatevecSamplerPreprocess(handle, sampler, extraWorkspace, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecSamplerPreprocess, libcustatevec), custatevecStatus_t, (custatevecHandle_t, custatevecSamplerDescriptor_t, CuPtr{Cvoid}, Csize_t), handle, sampler, extraWorkspace, extraWorkspaceSizeInBytes)
end

function custatevecComputeExpectation(handle, sv, svDataType, nIndexBits, expectationValue, expectationDataType, residualNorm, matrix, matrixDataType, layout, basisBits, nBasisBits, computeType, extraWorkspace, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecComputeExpectation, libcustatevec), custatevecStatus_t, (custatevecHandle_t, CuPtr{Cvoid}, cudaDataType_t, UInt32, Ptr{Cvoid}, cudaDataType_t, Ptr{Cdouble}, PtrOrCuPtr{Cvoid}, cudaDataType_t, custatevecMatrixLayout_t, Ptr{Int32}, UInt32, custatevecComputeType_t, CuPtr{Cvoid}, Csize_t), handle, sv, svDataType, nIndexBits, expectationValue, expectationDataType, residualNorm, matrix, matrixDataType, layout, basisBits, nBasisBits, computeType, extraWorkspace, extraWorkspaceSizeInBytes)
end

function custatevecSamplerCreate(handle, sv, svDataType, nIndexBits, sampler, nMaxShots, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecSamplerCreate, libcustatevec), custatevecStatus_t, (custatevecHandle_t, CuPtr{Cvoid}, cudaDataType_t, UInt32, Ptr{custatevecSamplerDescriptor_t}, UInt32, Ptr{Csize_t}), handle, sv, svDataType, nIndexBits, sampler, nMaxShots, extraWorkspaceSizeInBytes)
end

function custatevecApplyPauliRotation(handle, sv, svDataType, nIndexBits, theta, paulis, targets, nTargets, controls, controlBitValues, nControls)
    initialize_context()
    ccall((:custatevecApplyPauliRotation, libcustatevec), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Cdouble, Ptr{custatevecPauli_t}, Ptr{Int32}, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32), handle, sv, svDataType, nIndexBits, theta, paulis, targets, nTargets, controls, controlBitValues, nControls)
end

function custatevecGetDeviceMemHandler(handle, handler)
    initialize_context()
    ccall((:custatevecGetDeviceMemHandler, libcustatevec), custatevecStatus_t, (custatevecHandle_t, Ptr{custatevecDeviceMemHandler_t}), handle, handler)
end

function custatevecSamplerApplySubSVOffset(handle, sampler, subSVOrd, nSubSVs, offset, norm)
    initialize_context()
    ccall((:custatevecSamplerApplySubSVOffset, libcustatevec), custatevecStatus_t, (custatevecHandle_t, custatevecSamplerDescriptor_t, Int32, UInt32, Cdouble, Cdouble), handle, sampler, subSVOrd, nSubSVs, offset, norm)
end

function custatevecAccessorCreateView(handle, sv, svDataType, nIndexBits, accessor, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecAccessorCreateView, libcustatevec), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{custatevecAccessorDescriptor_t}, Ptr{Int32}, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32, Ptr{Csize_t}), handle, sv, svDataType, nIndexBits, accessor, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen, extraWorkspaceSizeInBytes)
end

function custatevecComputeExpectationsOnPauliBasis(handle, sv, svDataType, nIndexBits, expectationValues, pauliOperatorsArray, nPauliOperatorArrays, basisBitsArray, nBasisBitsArray)
    initialize_context()
    ccall((:custatevecComputeExpectationsOnPauliBasis, libcustatevec), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Cdouble}, Ptr{Ptr{custatevecPauli_t}}, UInt32, Ptr{Ptr{Int32}}, Ptr{UInt32}), handle, sv, svDataType, nIndexBits, expectationValues, pauliOperatorsArray, nPauliOperatorArrays, basisBitsArray, nBasisBitsArray)
end

function custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(handle, svDataType, nIndexBits, permutation, diagonals, diagonalsDataType, targets, nTargets, nControls, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize, libcustatevec), custatevecStatus_t, (custatevecHandle_t, cudaDataType_t, UInt32, Ptr{custatevecIndex_t}, Ptr{Cvoid}, cudaDataType_t, Ptr{Int32}, UInt32, UInt32, Ptr{Csize_t}), handle, svDataType, nIndexBits, permutation, diagonals, diagonalsDataType, targets, nTargets, nControls, extraWorkspaceSizeInBytes)
end

function custatevecApplyMatrix(handle, sv, svDataType, nIndexBits, matrix, matrixDataType, layout, adjoint, targets, nTargets, controls, controlBitValues, nControls, computeType, extraWorkspace, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecApplyMatrix, libcustatevec), custatevecStatus_t, (custatevecHandle_t, CuPtr{Cvoid}, cudaDataType_t, UInt32, PtrOrCuPtr{Cvoid}, cudaDataType_t, custatevecMatrixLayout_t, Int32, Ptr{Int32}, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32, custatevecComputeType_t, CuPtr{Cvoid}, Csize_t), handle, sv, svDataType, nIndexBits, matrix, matrixDataType, layout, adjoint, targets, nTargets, controls, controlBitValues, nControls, computeType, extraWorkspace, extraWorkspaceSizeInBytes)
end

function custatevecApplyGeneralizedPermutationMatrix(handle, sv, svDataType, nIndexBits, permutation, diagonals, diagonalsDataType, adjoint, targets, nTargets, controls, controlBitValues, nControls, extraWorkspace, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecApplyGeneralizedPermutationMatrix, libcustatevec), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{custatevecIndex_t}, Ptr{Cvoid}, cudaDataType_t, Int32, Ptr{Int32}, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32, Ptr{Cvoid}, Csize_t), handle, sv, svDataType, nIndexBits, permutation, diagonals, diagonalsDataType, adjoint, targets, nTargets, controls, controlBitValues, nControls, extraWorkspace, extraWorkspaceSizeInBytes)
end
