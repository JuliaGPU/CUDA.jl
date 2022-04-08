
function custatevecApplyGeneralizedPermutationMatrix(handle, sv, svDataType, nIndexBits, permutation, diagonals, diagonalsDataType, adjoint, basisBits, nBasisBits, maskBitString, maskOrdering, maskLen, extraWorkspace, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecApplyGeneralizedPermutationMatrix, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{custatevecIndex_t}, Ptr{Cvoid}, cudaDataType_t, Int32, Ptr{Int32}, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32, Ptr{Cvoid}, Csize_t), handle, sv, svDataType, nIndexBits, permutation, diagonals, diagonalsDataType, adjoint, basisBits, nBasisBits, maskBitString, maskOrdering, maskLen, extraWorkspace, extraWorkspaceSizeInBytes)
end

function custatevecGetErrorName(status)
    initialize_context()
    ccall((:custatevecGetErrorName, libcustatevec()), Cstring, (custatevecStatus_t,), status)
end

function custatevecAccessor_create(handle, sv, svDataType, nIndexBits, accessor, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecAccessor_create, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{custatevecAccessorDescriptor_t}, Ptr{Int32}, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32, Ptr{Csize_t}), handle, sv, svDataType, nIndexBits, accessor, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen, extraWorkspaceSizeInBytes)
end

function custatevecCreate(handle)
    initialize_context()
    ccall((:custatevecCreate, libcustatevec()), custatevecStatus_t, (Ptr{custatevecHandle_t},), handle)
end

function custatevecExpectationsOnPauliBasis(handle, sv, svDataType, nIndexBits, expectationValues, pauliOperatorsArray, basisBitsArray, nBasisBitsArray, nPauliOperatorArrays)
    initialize_context()
    ccall((:custatevecExpectationsOnPauliBasis, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Cdouble}, Ptr{Ptr{custatevecPauli_t}}, Ptr{Ptr{Int32}}, Ptr{UInt32}, UInt32), handle, sv, svDataType, nIndexBits, expectationValues, pauliOperatorsArray, basisBitsArray, nBasisBitsArray, nPauliOperatorArrays)
end

function custatevecApplyMatrix(handle, sv, svDataType, nIndexBits, matrix, matrixDataType, layout, adjoint, targets, nTargets, controls, nControls, controlBitValues, computeType, extraWorkspace, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecApplyMatrix, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, CuPtr{Cvoid}, cudaDataType_t, UInt32, PtrOrCuPtr{Cvoid}, cudaDataType_t, custatevecMatrixLayout_t, Int32, Ptr{Int32}, UInt32, Ptr{Int32}, UInt32, Ptr{Int32}, custatevecComputeType_t, CuPtr{Cvoid}, Csize_t), handle, sv, svDataType, nIndexBits, matrix, matrixDataType, layout, adjoint, targets, nTargets, controls, nControls, controlBitValues, computeType, extraWorkspace, extraWorkspaceSizeInBytes)
end

function custatevecAccessor_setExtraWorkspace(handle, accessor, extraWorkspace, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecAccessor_setExtraWorkspace, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, Ptr{custatevecAccessorDescriptor_t}, Ptr{Cvoid}, Csize_t), handle, accessor, extraWorkspace, extraWorkspaceSizeInBytes)
end

function custatevecSampler_create(handle, sv, svDataType, nIndexBits, sampler, nMaxShots, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecSampler_create, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, CuPtr{Cvoid}, cudaDataType_t, UInt32, Ptr{custatevecSamplerDescriptor_t}, UInt32, Ptr{Csize_t}), handle, sv, svDataType, nIndexBits, sampler, nMaxShots, extraWorkspaceSizeInBytes)
end

function custatevecCollapseOnZBasis(handle, sv, svDataType, nIndexBits, parity, basisBits, nBasisBits, norm)
    initialize_context()
    ccall((:custatevecCollapseOnZBasis, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Int32, Ptr{Int32}, UInt32, Cdouble), handle, sv, svDataType, nIndexBits, parity, basisBits, nBasisBits, norm)
end

function custatevecExpectation(handle, sv, svDataType, nIndexBits, expectationValue, expectationDataType, residualNorm, matrix, matrixDataType, layout, basisBits, nBasisBits, computeType, extraWorkspace, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecExpectation, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, PtrOrCuPtr{Cvoid}, cudaDataType_t, UInt32, Ptr{Cvoid}, cudaDataType_t, Ptr{Cdouble}, Ptr{Cvoid}, cudaDataType_t, custatevecMatrixLayout_t, Ptr{Int32}, UInt32, custatevecComputeType_t, CuPtr{Cvoid}, Csize_t), handle, sv, svDataType, nIndexBits, expectationValue, expectationDataType, residualNorm, matrix, matrixDataType, layout, basisBits, nBasisBits, computeType, extraWorkspace, extraWorkspaceSizeInBytes)
end

function custatevecBatchMeasure(handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, randnum, collapse)
    initialize_context()
    ccall((:custatevecBatchMeasure, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32, Cdouble, custatevecCollapseOp_t), handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, randnum, collapse)
end

function custatevecExpectation_bufferSize(handle, svDataType, nIndexBits, matrix, matrixDataType, layout, nBasisBits, computeType, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecExpectation_bufferSize, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, cudaDataType_t, UInt32, Ptr{Cvoid}, cudaDataType_t, custatevecMatrixLayout_t, UInt32, custatevecComputeType_t, Ptr{Csize_t}), handle, svDataType, nIndexBits, matrix, matrixDataType, layout, nBasisBits, computeType, extraWorkspaceSizeInBytes)
end

function custatevecApplyMatrix_bufferSize(handle, svDataType, nIndexBits, matrix, matrixDataType, layout, adjoint, nTargets, nControls, computeType, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecApplyMatrix_bufferSize, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, cudaDataType_t, UInt32, Ptr{Cvoid}, cudaDataType_t, custatevecMatrixLayout_t, Int32, UInt32, UInt32, custatevecComputeType_t, Ptr{Csize_t}), handle, svDataType, nIndexBits, matrix, matrixDataType, layout, adjoint, nTargets, nControls, computeType, extraWorkspaceSizeInBytes)
end

function custatevecLoggerSetMask(mask)
    ccall((:custatevecLoggerSetMask, libcustatevec()), custatevecStatus_t, (Int32,), mask)
end

function custatevecAccessor_createReadOnly(handle, sv, svDataType, nIndexBits, accessor, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecAccessor_createReadOnly, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{custatevecAccessorDescriptor_t}, Ptr{Int32}, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32, Ptr{Csize_t}), handle, sv, svDataType, nIndexBits, accessor, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen, extraWorkspaceSizeInBytes)
end

function custatevecGetStream(handle, streamId)
    initialize_context()
    ccall((:custatevecGetStream, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, Ptr{CUstream}), handle, streamId)
end

function custatevecAccessor_get(handle, accessor, externalBuffer, _begin, _end)
    initialize_context()
    ccall((:custatevecAccessor_get, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, Ptr{custatevecAccessorDescriptor_t}, Ptr{Cvoid}, custatevecIndex_t, custatevecIndex_t), handle, accessor, externalBuffer, _begin, _end)
end

function custatevecGetVersion()
    ccall((:custatevecGetVersion, libcustatevec()), Csize_t, ())
end

function custatevecLoggerForceDisable()
    ccall((:custatevecLoggerForceDisable, libcustatevec()), custatevecStatus_t, ())
end

function custatevecGetDefaultWorkspaceSize(handle, workspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecGetDefaultWorkspaceSize, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, Ptr{Csize_t}), handle, workspaceSizeInBytes)
end

function custatevecSetStream(handle, streamId)
    initialize_context()
    ccall((:custatevecSetStream, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, CUstream), handle, streamId)
end

function custatevecSetWorkspace(handle, workspace, workspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecSetWorkspace, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, Csize_t), handle, workspace, workspaceSizeInBytes)
end

function custatevecApplyExp(handle, sv, svDataType, nIndexBits, theta, paulis, targets, nTargets, controls, controlBitValues, nControls)
    initialize_context()
    ccall((:custatevecApplyExp, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Cdouble, Ptr{custatevecPauli_t}, Ptr{Int32}, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32), handle, sv, svDataType, nIndexBits, theta, paulis, targets, nTargets, controls, controlBitValues, nControls)
end

function custatevecAbs2SumArray(handle, sv, svDataType, nIndexBits, abs2sum, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen)
    initialize_context()
    ccall((:custatevecAbs2SumArray, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Cdouble}, Ptr{Int32}, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32), handle, sv, svDataType, nIndexBits, abs2sum, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen)
end

function custatevecGetProperty(type, value)
    initialize_context()
    ccall((:custatevecGetProperty, libcustatevec()), custatevecStatus_t, (libraryPropertyType, Ptr{Int32}), type, value)
end

function custatevecAbs2SumOnZBasis(handle, sv, svDataType, nIndexBits, abs2sum0, abs2sum1, basisBits, nBasisBits)
    initialize_context()
    ccall((:custatevecAbs2SumOnZBasis, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int32}, UInt32), handle, sv, svDataType, nIndexBits, abs2sum0, abs2sum1, basisBits, nBasisBits)
end

function custatevecLoggerOpenFile(logFile)
    ccall((:custatevecLoggerOpenFile, libcustatevec()), custatevecStatus_t, (Cstring,), logFile)
end

function custatevecDestroy(handle)
    initialize_context()
    ccall((:custatevecDestroy, libcustatevec()), custatevecStatus_t, (custatevecHandle_t,), handle)
end

function custatevecLoggerSetLevel(level)
    initialize_context()
    ccall((:custatevecLoggerSetLevel, libcustatevec()), custatevecStatus_t, (Int32,), level)
end

function custatevecApplyGeneralizedPermutationMatrix_bufferSize(handle, svDataType, nIndexBits, permutation, diagonals, diagonalsDataType, basisBits, nBasisBits, maskLen, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecApplyGeneralizedPermutationMatrix_bufferSize, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, cudaDataType_t, UInt32, Ptr{custatevecIndex_t}, Ptr{Cvoid}, cudaDataType_t, Ptr{Int32}, UInt32, UInt32, Ptr{Csize_t}), handle, svDataType, nIndexBits, permutation, diagonals, diagonalsDataType, basisBits, nBasisBits, maskLen, extraWorkspaceSizeInBytes)
end

function custatevecLoggerSetFile(file)
    ccall((:custatevecLoggerSetFile, libcustatevec()), custatevecStatus_t, (Ptr{Cvoid},), file)
end

function custatevecGetErrorString(status)
    ccall((:custatevecGetErrorString, libcustatevec()), Cstring, (custatevecStatus_t,), status)
end

function custatevecLoggerSetCallback(callback)
    ccall((:custatevecLoggerSetCallback, libcustatevec()), custatevecStatus_t, (custatevecLoggerCallback_t,), callback)
end

function custatevecCollapseByBitString(handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, norm)
    initialize_context()
    ccall((:custatevecCollapseByBitString, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32, Cdouble), handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, norm)
end

function custatevecSampler_preprocess(handle, sampler, extraWorkspace, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecSampler_preprocess, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, Ptr{custatevecSamplerDescriptor_t}, CuPtr{Cvoid}, Csize_t), handle, sampler, extraWorkspace, extraWorkspaceSizeInBytes)
end

function custatevecMeasureOnZBasis(handle, sv, svDataType, nIndexBits, parity, basisBits, nBasisBits, randnum, collapse)
    initialize_context()
    ccall((:custatevecMeasureOnZBasis, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32, Cdouble, custatevecCollapseOp_t), handle, sv, svDataType, nIndexBits, parity, basisBits, nBasisBits, randnum, collapse)
end

function custatevecSampler_sample(handle, sampler, bitStrings, bitOrdering, bitStringLen, randnums, nShots, output)
    initialize_context()
    ccall((:custatevecSampler_sample, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, Ptr{custatevecSamplerDescriptor_t}, Ptr{custatevecIndex_t}, Ptr{Int32}, UInt32, Ptr{Cdouble}, UInt32, custatevecSamplerOutput_t), handle, sampler, bitStrings, bitOrdering, bitStringLen, randnums, nShots, output)
end

function custatevecAccessor_set(handle, accessor, externalBuffer, _begin, _end)
    initialize_context()
    ccall((:custatevecAccessor_set, libcustatevec()), custatevecStatus_t, (custatevecHandle_t, Ptr{custatevecAccessorDescriptor_t}, Ptr{Cvoid}, custatevecIndex_t, custatevecIndex_t), handle, accessor, externalBuffer, _begin, _end)
end
