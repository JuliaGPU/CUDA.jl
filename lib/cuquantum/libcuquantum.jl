
function cutensornetGetCudartVersion()
    ccall((:cutensornetGetCudartVersion, libcuquantum()), Csize_t, ())
end

@checked function cutensornetContraction(handle, plan, rawDataIn, rawDataOut, workspace, workspaceSize, sliceId, stream)
    initialize_context()
    ccall((:cutensornetContraction, libcuquantum()), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionPlan_t, Ptr{Ptr{Cvoid}}, Ptr{Cvoid}, Ptr{Cvoid}, UInt64, Int64, CUstream), handle, plan, rawDataIn, rawDataOut, workspace, workspaceSize, sliceId, stream)
end

@checked function cutensornetCreateContractionOptimizerConfig(handle, optimizerConfig)
    initialize_context()
    ccall((:cutensornetCreateContractionOptimizerConfig, libcuquantum()), cutensornetStatus_t, (cutensornetHandle_t, Ptr{cutensornetContractionOptimizerConfig_t}), handle, optimizerConfig)
end

@checked function cutensornetCreate(handle)
    initialize_context()
    ccall((:cutensornetCreate, libcuquantum()), cutensornetStatus_t, (Ptr{cutensornetHandle_t},), handle)
end

@checked function cutensornetContractionGetWorkspaceSize(handle, descNet, optimizerInfo, workspaceSize)
    initialize_context()
    ccall((:cutensornetContractionGetWorkspaceSize, libcuquantum()), cutensornetStatus_t, (cutensornetHandle_t, cutensornetNetworkDescriptor_t, cutensornetContractionOptimizerInfo_t, Ptr{UInt64}), handle, descNet, optimizerInfo, workspaceSize)
end

@checked function cutensornetDestroyContractionOptimizerInfo(optimizerInfo)
    initialize_context()
    ccall((:cutensornetDestroyContractionOptimizerInfo, libcuquantum()), cutensornetStatus_t, (cutensornetContractionOptimizerInfo_t,), optimizerInfo)
end

@checked function cutensornetLoggerOpenFile(logFile)
    initialize_context()
    ccall((:cutensornetLoggerOpenFile, libcuquantum()), cutensornetStatus_t, (Cstring,), logFile)
end

@checked function cutensornetContractionAutotunePreferenceSetAttribute(handle, autotunePreference, attr, buf, sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionAutotunePreferenceSetAttribute, libcuquantum()), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionAutotunePreference_t, cutensornetContractionAutotunePreferenceAttributes_t, Ptr{Cvoid}, Csize_t), handle, autotunePreference, attr, buf, sizeInBytes)
end

@checked function cutensornetLoggerSetCallback(callback)
    initialize_context()
    ccall((:cutensornetLoggerSetCallback, libcuquantum()), cutensornetStatus_t, (cutensornetLoggerCallback_t,), callback)
end

@checked function cutensornetLoggerForceDisable()
    initialize_context()
    ccall((:cutensornetLoggerForceDisable, libcuquantum()), cutensornetStatus_t, ())
end

@checked function cutensornetLoggerSetFile(file)
    initialize_context()
    ccall((:cutensornetLoggerSetFile, libcuquantum()), cutensornetStatus_t, (Ptr{FILE},), file)
end

@checked function cutensornetContractionOptimize(handle, descNet, optimizerConfig, workspaceSizeConstraint, optimizerInfo)
    initialize_context()
    ccall((:cutensornetContractionOptimize, libcuquantum()), cutensornetStatus_t, (cutensornetHandle_t, cutensornetNetworkDescriptor_t, cutensornetContractionOptimizerConfig_t, UInt64, cutensornetContractionOptimizerInfo_t), handle, descNet, optimizerConfig, workspaceSizeConstraint, optimizerInfo)
end

@checked function cutensornetCreateContractionAutotunePreference(handle, autotunePreference)
    initialize_context()
    ccall((:cutensornetCreateContractionAutotunePreference, libcuquantum()), cutensornetStatus_t, (cutensornetHandle_t, Ptr{cutensornetContractionAutotunePreference_t}), handle, autotunePreference)
end

@checked function cutensornetCreateNetworkDescriptor(handle, numInputs, numModesIn, extentsIn, stridesIn, modesIn, alignmentRequirementsIn, numModesOut, extentsOut, stridesOut, modesOut, alignmentRequirementsOut, dataType, computeType, descNet)
    initialize_context()
    ccall((:cutensornetCreateNetworkDescriptor, libcuquantum()), cutensornetStatus_t, (cutensornetHandle_t, Int32, Ptr{Int32}, Ptr{Ptr{Int64}}, Ptr{Ptr{Int64}}, Ptr{Ptr{Int32}}, Ptr{UInt32}, Int32, Ptr{Int64}, Ptr{Int64}, Ptr{Int32}, UInt32, cudaDataType_t, cutensornetComputeType_t, Ptr{cutensornetNetworkDescriptor_t}), handle, numInputs, numModesIn, extentsIn, stridesIn, modesIn, alignmentRequirementsIn, numModesOut, extentsOut, stridesOut, modesOut, alignmentRequirementsOut, dataType, computeType, descNet)
end

@checked function cutensornetContractionOptimizerConfigGetAttribute(handle, optimizerConfig, attr, buf, sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionOptimizerConfigGetAttribute, libcuquantum()), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionOptimizerConfig_t, cutensornetContractionOptimizerConfigAttributes_t, Ptr{Cvoid}, Csize_t), handle, optimizerConfig, attr, buf, sizeInBytes)
end

@checked function cutensornetContractionOptimizerInfoGetAttribute(handle, optimizerInfo, attr, buf, sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionOptimizerInfoGetAttribute, libcuquantum()), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionOptimizerInfo_t, cutensornetContractionOptimizerInfoAttributes_t, Ptr{Cvoid}, Csize_t), handle, optimizerInfo, attr, buf, sizeInBytes)
end

@checked function cutensornetContractionOptimizerInfoSetAttribute(handle, optimizerInfo, attr, buf, sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionOptimizerInfoSetAttribute, libcuquantum()), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionOptimizerInfo_t, cutensornetContractionOptimizerInfoAttributes_t, Ptr{Cvoid}, Csize_t), handle, optimizerInfo, attr, buf, sizeInBytes)
end

@checked function cutensornetDestroyContractionPlan(plan)
    initialize_context()
    ccall((:cutensornetDestroyContractionPlan, libcuquantum()), cutensornetStatus_t, (cutensornetContractionPlan_t,), plan)
end

@checked function cutensornetLoggerSetLevel(level)
    initialize_context()
    ccall((:cutensornetLoggerSetLevel, libcuquantum()), cutensornetStatus_t, (Int32,), level)
end

@checked function cutensornetDestroy(handle)
    initialize_context()
    ccall((:cutensornetDestroy, libcuquantum()), cutensornetStatus_t, (cutensornetHandle_t,), handle)
end

@checked function cutensornetDestroyNetworkDescriptor(desc)
    initialize_context()
    ccall((:cutensornetDestroyNetworkDescriptor, libcuquantum()), cutensornetStatus_t, (cutensornetNetworkDescriptor_t,), desc)
end

@checked function cutensornetContractionAutotune(handle, plan, rawDataIn, rawDataOut, workspace, workspaceSize, pref, stream)
    initialize_context()
    ccall((:cutensornetContractionAutotune, libcuquantum()), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionPlan_t, Ptr{Ptr{Cvoid}}, Ptr{Cvoid}, Ptr{Cvoid}, UInt64, cutensornetContractionAutotunePreference_t, CUstream), handle, plan, rawDataIn, rawDataOut, workspace, workspaceSize, pref, stream)
end

@checked function cutensornetDestroyContractionAutotunePreference(autotunePreference)
    initialize_context()
    ccall((:cutensornetDestroyContractionAutotunePreference, libcuquantum()), cutensornetStatus_t, (cutensornetContractionAutotunePreference_t,), autotunePreference)
end

@checked function cutensornetCreateContractionOptimizerInfo(handle, descNet, optimizerInfo)
    initialize_context()
    ccall((:cutensornetCreateContractionOptimizerInfo, libcuquantum()), cutensornetStatus_t, (cutensornetHandle_t, cutensornetNetworkDescriptor_t, Ptr{cutensornetContractionOptimizerInfo_t}), handle, descNet, optimizerInfo)
end

function cutensornetGetVersion()
    ccall((:cutensornetGetVersion, libcuquantum()), Csize_t, ())
end

function cutensornetGetErrorString(error)
    ccall((:cutensornetGetErrorString, libcuquantum()), Cstring, (cutensornetStatus_t,), error)
end

@checked function cutensornetDestroyContractionOptimizerConfig(optimizerConfig)
    initialize_context()
    ccall((:cutensornetDestroyContractionOptimizerConfig, libcuquantum()), cutensornetStatus_t, (cutensornetContractionOptimizerConfig_t,), optimizerConfig)
end

@checked function cutensornetContractionOptimizerConfigSetAttribute(handle, optimizerConfig, attr, buf, sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionOptimizerConfigSetAttribute, libcuquantum()), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionOptimizerConfig_t, cutensornetContractionOptimizerConfigAttributes_t, Ptr{Cvoid}, Csize_t), handle, optimizerConfig, attr, buf, sizeInBytes)
end

@checked function cutensornetLoggerSetMask(mask)
    initialize_context()
    ccall((:cutensornetLoggerSetMask, libcuquantum()), cutensornetStatus_t, (Int32,), mask)
end

@checked function cutensornetContractionAutotunePreferenceGetAttribute(handle, autotunePreference, attr, buf, sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionAutotunePreferenceGetAttribute, libcuquantum()), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionAutotunePreference_t, cutensornetContractionAutotunePreferenceAttributes_t, Ptr{Cvoid}, Csize_t), handle, autotunePreference, attr, buf, sizeInBytes)
end

@checked function cutensornetCreateContractionPlan(handle, descNet, optimizerInfo, workspaceSize, plan)
    initialize_context()
    ccall((:cutensornetCreateContractionPlan, libcuquantum()), cutensornetStatus_t, (cutensornetHandle_t, cutensornetNetworkDescriptor_t, cutensornetContractionOptimizerInfo_t, UInt64, Ptr{cutensornetContractionPlan_t}), handle, descNet, optimizerInfo, workspaceSize, plan)
end

function custatevecApplyGeneralizedPermutationMatrix(handle, sv, svDataType, nIndexBits, permutation, diagonals, diagonalsDataType, adjoint, basisBits, nBasisBits, maskBitString, maskOrdering, maskLen, extraWorkspace, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecApplyGeneralizedPermutationMatrix, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{custatevecIndex_t}, Ptr{Cvoid}, cudaDataType_t, Int32, Ptr{Int32}, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32, Ptr{Cvoid}, Csize_t), handle, sv, svDataType, nIndexBits, permutation, diagonals, diagonalsDataType, adjoint, basisBits, nBasisBits, maskBitString, maskOrdering, maskLen, extraWorkspace, extraWorkspaceSizeInBytes)
end

function custatevecAccessor_create(handle, sv, svDataType, nIndexBits, accessor, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecAccessor_create, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{custatevecAccessorDescriptor_t}, Ptr{Int32}, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32, Ptr{Csize_t}), handle, sv, svDataType, nIndexBits, accessor, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen, extraWorkspaceSizeInBytes)
end

function custatevecExpectationsOnPauliBasis(handle, sv, svDataType, nIndexBits, expectationValues, pauliOperatorsArray, basisBitsArray, nBasisBitsArray, nPauliOperatorArrays)
    initialize_context()
    ccall((:custatevecExpectationsOnPauliBasis, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Cdouble}, Ptr{Ptr{custatevecPauli_t}}, Ptr{Ptr{Int32}}, Ptr{UInt32}, UInt32), handle, sv, svDataType, nIndexBits, expectationValues, pauliOperatorsArray, basisBitsArray, nBasisBitsArray, nPauliOperatorArrays)
end

function custatevecSampler_create(handle, sv, svDataType, nIndexBits, sampler, nMaxShots, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecSampler_create, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{custatevecSamplerDescriptor_t}, UInt32, Ptr{Csize_t}), handle, sv, svDataType, nIndexBits, sampler, nMaxShots, extraWorkspaceSizeInBytes)
end

function custatevecExpectation(handle, sv, svDataType, nIndexBits, expectationValue, expectationDataType, residualNorm, matrix, matrixDataType, layout, basisBits, nBasisBits, computeType, extraWorkspace, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecExpectation, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Cvoid}, cudaDataType_t, Ptr{Cdouble}, Ptr{Cvoid}, cudaDataType_t, custatevecMatrixLayout_t, Ptr{Int32}, UInt32, custatevecComputeType_t, Ptr{Cvoid}, Csize_t), handle, sv, svDataType, nIndexBits, expectationValue, expectationDataType, residualNorm, matrix, matrixDataType, layout, basisBits, nBasisBits, computeType, extraWorkspace, extraWorkspaceSizeInBytes)
end

function custatevecBatchMeasure(handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, randnum, collapse)
    initialize_context()
    ccall((:custatevecBatchMeasure, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32, Cdouble, custatevecCollapseOp_t), handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, randnum, collapse)
end

function custatevecExpectation_bufferSize(handle, svDataType, nIndexBits, matrix, matrixDataType, layout, nBasisBits, computeType, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecExpectation_bufferSize, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, cudaDataType_t, UInt32, Ptr{Cvoid}, cudaDataType_t, custatevecMatrixLayout_t, UInt32, custatevecComputeType_t, Ptr{Csize_t}), handle, svDataType, nIndexBits, matrix, matrixDataType, layout, nBasisBits, computeType, extraWorkspaceSizeInBytes)
end

function custatevecApplyMatrix_bufferSize(handle, svDataType, nIndexBits, matrix, matrixDataType, layout, adjoint, nTargets, nControls, computeType, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecApplyMatrix_bufferSize, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, cudaDataType_t, UInt32, Ptr{Cvoid}, cudaDataType_t, custatevecMatrixLayout_t, Int32, UInt32, UInt32, custatevecComputeType_t, Ptr{Csize_t}), handle, svDataType, nIndexBits, matrix, matrixDataType, layout, adjoint, nTargets, nControls, computeType, extraWorkspaceSizeInBytes)
end

function custatevecAccessor_createReadOnly(handle, sv, svDataType, nIndexBits, accessor, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecAccessor_createReadOnly, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{custatevecAccessorDescriptor_t}, Ptr{Int32}, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32, Ptr{Csize_t}), handle, sv, svDataType, nIndexBits, accessor, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen, extraWorkspaceSizeInBytes)
end

function custatevecGetStream(handle, streamId)
    initialize_context()
    ccall((:custatevecGetStream, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{CUstream}), handle, streamId)
end

function custatevecAccessor_get(handle, accessor, externalBuffer, _begin, _end)
    initialize_context()
    ccall((:custatevecAccessor_get, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{custatevecAccessorDescriptor_t}, Ptr{Cvoid}, custatevecIndex_t, custatevecIndex_t), handle, accessor, externalBuffer, _begin, _end)
end

function custatevecGetVersion()
    ccall((:custatevecGetVersion, libcuquantum()), Csize_t, ())
end

function custatevecLoggerForceDisable()
    ccall((:custatevecLoggerForceDisable, libcuquantum()), custatevecStatus_t, ())
end

function custatevecSetWorkspace(handle, workspace, workspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecSetWorkspace, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, Csize_t), handle, workspace, workspaceSizeInBytes)
end

function custatevecApplyExp(handle, sv, svDataType, nIndexBits, theta, paulis, targets, nTargets, controls, controlBitValues, nControls)
    initialize_context()
    ccall((:custatevecApplyExp, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Cdouble, Ptr{custatevecPauli_t}, Ptr{Int32}, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32), handle, sv, svDataType, nIndexBits, theta, paulis, targets, nTargets, controls, controlBitValues, nControls)
end

function custatevecGetProperty(type, value)
    initialize_context()
    ccall((:custatevecGetProperty, libcuquantum()), custatevecStatus_t, (libraryPropertyType, Ptr{Int32}), type, value)
end

function custatevecAbs2SumOnZBasis(handle, sv, svDataType, nIndexBits, abs2sum0, abs2sum1, basisBits, nBasisBits)
    initialize_context()
    ccall((:custatevecAbs2SumOnZBasis, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Int32}, UInt32), handle, sv, svDataType, nIndexBits, abs2sum0, abs2sum1, basisBits, nBasisBits)
end

function custatevecDestroy(handle)
    initialize_context()
    ccall((:custatevecDestroy, libcuquantum()), custatevecStatus_t, (custatevecHandle_t,), handle)
end

function custatevecLoggerSetLevel(level)
    initialize_context()
    ccall((:custatevecLoggerSetLevel, libcuquantum()), custatevecStatus_t, (Int32,), level)
end

function custatevecApplyGeneralizedPermutationMatrix_bufferSize(handle, svDataType, nIndexBits, permutation, diagonals, diagonalsDataType, basisBits, nBasisBits, maskLen, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecApplyGeneralizedPermutationMatrix_bufferSize, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, cudaDataType_t, UInt32, Ptr{custatevecIndex_t}, Ptr{Cvoid}, cudaDataType_t, Ptr{Int32}, UInt32, UInt32, Ptr{Csize_t}), handle, svDataType, nIndexBits, permutation, diagonals, diagonalsDataType, basisBits, nBasisBits, maskLen, extraWorkspaceSizeInBytes)
end

function custatevecCollapseByBitString(handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, norm)
    initialize_context()
    ccall((:custatevecCollapseByBitString, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32, Cdouble), handle, sv, svDataType, nIndexBits, bitString, bitOrdering, bitStringLen, norm)
end

function custatevecSampler_sample(handle, sampler, bitStrings, bitOrdering, bitStringLen, randnums, nShots, output)
    initialize_context()
    ccall((:custatevecSampler_sample, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{custatevecSamplerDescriptor_t}, Ptr{custatevecIndex_t}, Ptr{Int32}, UInt32, Ptr{Cdouble}, UInt32, custatevecSamplerOutput_t), handle, sampler, bitStrings, bitOrdering, bitStringLen, randnums, nShots, output)
end

function custatevecGetErrorName(status)
    initialize_context()
    ccall((:custatevecGetErrorName, libcuquantum()), Cstring, (custatevecStatus_t,), status)
end

function custatevecCreate(handle)
    initialize_context()
    ccall((:custatevecCreate, libcuquantum()), custatevecStatus_t, (Ptr{custatevecHandle_t},), handle)
end

function custatevecAccessor_setExtraWorkspace(handle, accessor, extraWorkspace, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecAccessor_setExtraWorkspace, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{custatevecAccessorDescriptor_t}, Ptr{Cvoid}, Csize_t), handle, accessor, extraWorkspace, extraWorkspaceSizeInBytes)
end

function custatevecApplyMatrix(handle, sv, svDataType, nIndexBits, matrix, matrixDataType, layout, adjoint, targets, nTargets, controls, nControls, controlBitValues, computeType, extraWorkspace, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecApplyMatrix, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Cvoid}, cudaDataType_t, custatevecMatrixLayout_t, Int32, Ptr{Int32}, UInt32, Ptr{Int32}, UInt32, Ptr{Int32}, custatevecComputeType_t, Ptr{Cvoid}, Csize_t), handle, sv, svDataType, nIndexBits, matrix, matrixDataType, layout, adjoint, targets, nTargets, controls, nControls, controlBitValues, computeType, extraWorkspace, extraWorkspaceSizeInBytes)
end

function custatevecCollapseOnZBasis(handle, sv, svDataType, nIndexBits, parity, basisBits, nBasisBits, norm)
    initialize_context()
    ccall((:custatevecCollapseOnZBasis, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Int32, Ptr{Int32}, UInt32, Cdouble), handle, sv, svDataType, nIndexBits, parity, basisBits, nBasisBits, norm)
end

function custatevecLoggerSetMask(mask)
    ccall((:custatevecLoggerSetMask, libcuquantum()), custatevecStatus_t, (Int32,), mask)
end

function custatevecSetStream(handle, streamId)
    initialize_context()
    ccall((:custatevecSetStream, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, CUstream), handle, streamId)
end

function custatevecGetDefaultWorkspaceSize(handle, workspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecGetDefaultWorkspaceSize, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{Csize_t}), handle, workspaceSizeInBytes)
end

function custatevecAbs2SumArray(handle, sv, svDataType, nIndexBits, abs2sum, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen)
    initialize_context()
    ccall((:custatevecAbs2SumArray, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Cdouble}, Ptr{Int32}, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32), handle, sv, svDataType, nIndexBits, abs2sum, bitOrdering, bitOrderingLen, maskBitString, maskOrdering, maskLen)
end

function custatevecLoggerOpenFile(logFile)
    ccall((:custatevecLoggerOpenFile, libcuquantum()), custatevecStatus_t, (Cstring,), logFile)
end

function custatevecLoggerSetFile(file)
    ccall((:custatevecLoggerSetFile, libcuquantum()), custatevecStatus_t, (Ptr{FILE},), file)
end

function custatevecGetErrorString(status)
    ccall((:custatevecGetErrorString, libcuquantum()), Cstring, (custatevecStatus_t,), status)
end

function custatevecLoggerSetCallback(callback)
    ccall((:custatevecLoggerSetCallback, libcuquantum()), custatevecStatus_t, (custatevecLoggerCallback_t,), callback)
end

function custatevecSampler_preprocess(handle, sampler, extraWorkspace, extraWorkspaceSizeInBytes)
    initialize_context()
    ccall((:custatevecSampler_preprocess, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{custatevecSamplerDescriptor_t}, Ptr{Cvoid}, Csize_t), handle, sampler, extraWorkspace, extraWorkspaceSizeInBytes)
end

function custatevecMeasureOnZBasis(handle, sv, svDataType, nIndexBits, parity, basisBits, nBasisBits, randnum, collapse)
    initialize_context()
    ccall((:custatevecMeasureOnZBasis, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{Cvoid}, cudaDataType_t, UInt32, Ptr{Int32}, Ptr{Int32}, UInt32, Cdouble, custatevecCollapseOp_t), handle, sv, svDataType, nIndexBits, parity, basisBits, nBasisBits, randnum, collapse)
end

function custatevecAccessor_set(handle, accessor, externalBuffer, _begin, _end)
    initialize_context()
    ccall((:custatevecAccessor_set, libcuquantum()), custatevecStatus_t, (custatevecHandle_t, Ptr{custatevecAccessorDescriptor_t}, Ptr{Cvoid}, custatevecIndex_t, custatevecIndex_t), handle, accessor, externalBuffer, _begin, _end)
end
