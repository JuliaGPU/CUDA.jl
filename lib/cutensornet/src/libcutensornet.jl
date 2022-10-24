
function cutensornetGetCudartVersion()
    ccall((:cutensornetGetCudartVersion, libcutensornet), Csize_t, ())
end

@checked function cutensornetCreateContractionOptimizerConfig(handle, optimizerConfig)
    initialize_context()
    ccall((:cutensornetCreateContractionOptimizerConfig, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, Ptr{cutensornetContractionOptimizerConfig_t}), handle, optimizerConfig)
end

@checked function cutensornetCreate(handle)
    initialize_context()
    ccall((:cutensornetCreate, libcutensornet), cutensornetStatus_t, (Ptr{cutensornetHandle_t},), handle)
end

@checked function cutensornetDestroyContractionOptimizerInfo(optimizerInfo)
    initialize_context()
    ccall((:cutensornetDestroyContractionOptimizerInfo, libcutensornet), cutensornetStatus_t, (cutensornetContractionOptimizerInfo_t,), optimizerInfo)
end

@checked function cutensornetLoggerOpenFile(logFile)
    ccall((:cutensornetLoggerOpenFile, libcutensornet), cutensornetStatus_t, (Cstring,), logFile)
end

@checked function cutensornetLoggerSetCallback(callback)
    ccall((:cutensornetLoggerSetCallback, libcutensornet), cutensornetStatus_t, (cutensornetLoggerCallback_t,), callback)
end

@checked function cutensornetLoggerForceDisable()
    ccall((:cutensornetLoggerForceDisable, libcutensornet), cutensornetStatus_t, ())
end

@checked function cutensornetLoggerSetFile(file)
    ccall((:cutensornetLoggerSetFile, libcutensornet), cutensornetStatus_t, (Ptr{Cvoid},), file)
end

@checked function cutensornetContractionOptimize(handle, descNet, optimizerConfig, workspaceSizeConstraint, optimizerInfo)
    initialize_context()
    ccall((:cutensornetContractionOptimize, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetNetworkDescriptor_t, cutensornetContractionOptimizerConfig_t, UInt64, cutensornetContractionOptimizerInfo_t), handle, descNet, optimizerConfig, workspaceSizeConstraint, optimizerInfo)
end

@checked function cutensornetCreateContractionAutotunePreference(handle, autotunePreference)
    initialize_context()
    ccall((:cutensornetCreateContractionAutotunePreference, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, Ptr{cutensornetContractionAutotunePreference_t}), handle, autotunePreference)
end

@checked function cutensornetCreateNetworkDescriptor(handle, numInputs, numModesIn, extentsIn, stridesIn, modesIn, alignmentRequirementsIn, numModesOut, extentsOut, stridesOut, modesOut, alignmentRequirementsOut, dataType, computeType, descNet)
    initialize_context()
    ccall((:cutensornetCreateNetworkDescriptor, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, Int32, Ptr{Int32}, Ptr{Ptr{Int64}}, Ptr{Ptr{Int64}}, Ptr{Ptr{Int32}}, Ptr{UInt32}, Int32, Ptr{Int64}, Ptr{Int64}, Ptr{Int32}, UInt32, cudaDataType_t, cutensornetComputeType_t, Ptr{cutensornetNetworkDescriptor_t}), handle, numInputs, numModesIn, extentsIn, stridesIn, modesIn, alignmentRequirementsIn, numModesOut, extentsOut, stridesOut, modesOut, alignmentRequirementsOut, dataType, computeType, descNet)
end

@checked function cutensornetContractionOptimizerConfigGetAttribute(handle, optimizerConfig, attr, buf, sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionOptimizerConfigGetAttribute, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionOptimizerConfig_t, cutensornetContractionOptimizerConfigAttributes_t, Ptr{Cvoid}, Csize_t), handle, optimizerConfig, attr, buf, sizeInBytes)
end

@checked function cutensornetContractionOptimizerInfoGetAttribute(handle, optimizerInfo, attr, buf, sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionOptimizerInfoGetAttribute, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionOptimizerInfo_t, cutensornetContractionOptimizerInfoAttributes_t, Ptr{Cvoid}, Csize_t), handle, optimizerInfo, attr, buf, sizeInBytes)
end

@checked function cutensornetContractionOptimizerInfoSetAttribute(handle, optimizerInfo, attr, buf, sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionOptimizerInfoSetAttribute, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionOptimizerInfo_t, cutensornetContractionOptimizerInfoAttributes_t, Ptr{Cvoid}, Csize_t), handle, optimizerInfo, attr, buf, sizeInBytes)
end

@checked function cutensornetDestroyContractionPlan(plan)
    initialize_context()
    ccall((:cutensornetDestroyContractionPlan, libcutensornet), cutensornetStatus_t, (cutensornetContractionPlan_t,), plan)
end

@checked function cutensornetLoggerSetLevel(level)
    initialize_context()
    ccall((:cutensornetLoggerSetLevel, libcutensornet), cutensornetStatus_t, (Int32,), level)
end

@checked function cutensornetDestroy(handle)
    initialize_context()
    ccall((:cutensornetDestroy, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t,), handle)
end

@checked function cutensornetDestroyNetworkDescriptor(desc)
    initialize_context()
    ccall((:cutensornetDestroyNetworkDescriptor, libcutensornet), cutensornetStatus_t, (cutensornetNetworkDescriptor_t,), desc)
end

@checked function cutensornetDestroyContractionAutotunePreference(autotunePreference)
    initialize_context()
    ccall((:cutensornetDestroyContractionAutotunePreference, libcutensornet), cutensornetStatus_t, (cutensornetContractionAutotunePreference_t,), autotunePreference)
end

@checked function cutensornetCreateContractionOptimizerInfo(handle, descNet, optimizerInfo)
    initialize_context()
    ccall((:cutensornetCreateContractionOptimizerInfo, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetNetworkDescriptor_t, Ptr{cutensornetContractionOptimizerInfo_t}), handle, descNet, optimizerInfo)
end

function cutensornetGetVersion()
    ccall((:cutensornetGetVersion, libcutensornet), Csize_t, ())
end

function cutensornetGetErrorString(error)
    ccall((:cutensornetGetErrorString, libcutensornet), Cstring, (cutensornetStatus_t,), error)
end

@checked function cutensornetDestroyContractionOptimizerConfig(optimizerConfig)
    initialize_context()
    ccall((:cutensornetDestroyContractionOptimizerConfig, libcutensornet), cutensornetStatus_t, (cutensornetContractionOptimizerConfig_t,), optimizerConfig)
end

@checked function cutensornetContractionOptimizerConfigSetAttribute(handle, optimizerConfig, attr, buf, sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionOptimizerConfigSetAttribute, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionOptimizerConfig_t, cutensornetContractionOptimizerConfigAttributes_t, Ptr{Cvoid}, Csize_t), handle, optimizerConfig, attr, buf, sizeInBytes)
end

@checked function cutensornetLoggerSetMask(mask)
    ccall((:cutensornetLoggerSetMask, libcutensornet), cutensornetStatus_t, (Int32,), mask)
end

@checked function cutensornetContractionAutotunePreferenceGetAttribute(handle, autotunePreference, attr, buf, sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionAutotunePreferenceGetAttribute, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionAutotunePreference_t, cutensornetContractionAutotunePreferenceAttributes_t, Ptr{Cvoid}, Csize_t), handle, autotunePreference, attr, buf, sizeInBytes)
end

@checked function cutensornetContractionOptimizerInfoPackData(handle, optimizerInfo, buffer, sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionOptimizerInfoPackData, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionOptimizerInfo_t, Ptr{Cvoid}, Csize_t), handle, optimizerInfo, buffer, sizeInBytes)
end

@checked function cutensornetWorkspaceComputeSizes(handle, descNet, optimizerInfo, workDesc)
    initialize_context()
    ccall((:cutensornetWorkspaceComputeSizes, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetNetworkDescriptor_t, cutensornetContractionOptimizerInfo_t, cutensornetWorkspaceDescriptor_t), handle, descNet, optimizerInfo, workDesc)
end

@checked function cutensornetDestroyWorkspaceDescriptor(desc)
    initialize_context()
    ccall((:cutensornetDestroyWorkspaceDescriptor, libcutensornet), cutensornetStatus_t, (cutensornetWorkspaceDescriptor_t,), desc)
end

@checked function cutensornetWorkspaceSet(handle, workDesc, memSpace, workspacePtr, workspaceSize)
    initialize_context()
    ccall((:cutensornetWorkspaceSet, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetWorkspaceDescriptor_t, cutensornetMemspace_t, PtrOrCuPtr{Cvoid}, UInt64), handle, workDesc, memSpace, workspacePtr, workspaceSize)
end

@checked function cutensornetDestroySliceGroup(sliceGroup)
    initialize_context()
    ccall((:cutensornetDestroySliceGroup, libcutensornet), cutensornetStatus_t, (cutensornetSliceGroup_t,), sliceGroup)
end

@checked function cutensornetContractionOptimizerInfoGetPackedSize(handle, optimizerInfo, sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionOptimizerInfoGetPackedSize, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionOptimizerInfo_t, Ptr{Csize_t}), handle, optimizerInfo, sizeInBytes)
end

@checked function cutensornetCreateSliceGroupFromIDRange(handle, sliceIdStart, sliceIdStop, sliceIdStep, sliceGroup)
    initialize_context()
    ccall((:cutensornetCreateSliceGroupFromIDRange, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, Int64, Int64, Int64, Ptr{cutensornetSliceGroup_t}), handle, sliceIdStart, sliceIdStop, sliceIdStep, sliceGroup)
end

@checked function cutensornetWorkspaceGet(handle, workDesc, memSpace, workspacePtr, workspaceSize)
    initialize_context()
    ccall((:cutensornetWorkspaceGet, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetWorkspaceDescriptor_t, cutensornetMemspace_t, Ptr{Ptr{Cvoid}}, Ptr{UInt64}), handle, workDesc, memSpace, workspacePtr, workspaceSize)
end

@checked function cutensornetUpdateContractionOptimizerInfoFromPackedData(handle, buffer, sizeInBytes, optimizerInfo)
    initialize_context()
    ccall((:cutensornetUpdateContractionOptimizerInfoFromPackedData, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, Ptr{Cvoid}, Csize_t, cutensornetContractionOptimizerInfo_t), handle, buffer, sizeInBytes, optimizerInfo)
end

@checked function cutensornetLoggerSetCallbackData(callback, userData)
    initialize_context()
    ccall((:cutensornetLoggerSetCallbackData, libcutensornet), cutensornetStatus_t, (cutensornetLoggerCallbackData_t, Ptr{Cvoid}), callback, userData)
end

@checked function cutensornetGetDeviceMemHandler(handle, devMemHandler)
    initialize_context()
    ccall((:cutensornetGetDeviceMemHandler, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, Ptr{cutensornetDeviceMemHandler_t}), handle, devMemHandler)
end

@checked function cutensornetCreateWorkspaceDescriptor(handle, workDesc)
    initialize_context()
    ccall((:cutensornetCreateWorkspaceDescriptor, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, Ptr{cutensornetWorkspaceDescriptor_t}), handle, workDesc)
end

@checked function cutensornetGetOutputTensorDetails(handle, descNet, numModesOut, dataSizeOut, modeLabelsOut, extentsOut, stridesOut)
    initialize_context()
    ccall((:cutensornetGetOutputTensorDetails, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetNetworkDescriptor_t, Ptr{Int32}, Ptr{Csize_t}, Ptr{Int32}, Ptr{Int64}, Ptr{Int64}), handle, descNet, numModesOut, dataSizeOut, modeLabelsOut, extentsOut, stridesOut)
end

@checked function cutensornetContractSlices(handle, plan, rawDataIn, rawDataOut, accumulateOutput, workDesc, sliceGroup, stream)
    initialize_context()
    ccall((:cutensornetContractSlices, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionPlan_t, Ptr{Ptr{Cvoid}}, Ptr{Cvoid}, Int32, cutensornetWorkspaceDescriptor_t, cutensornetSliceGroup_t, CUstream), handle, plan, rawDataIn, rawDataOut, accumulateOutput, workDesc, sliceGroup, stream)
end

@checked function cutensornetWorkspaceGetSize(handle, workDesc, workPref, memSpace, workspaceSize)
    initialize_context()
    ccall((:cutensornetWorkspaceGetSize, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetWorkspaceDescriptor_t, cutensornetWorksizePref_t, cutensornetMemspace_t, Ptr{UInt64}), handle, workDesc, workPref, memSpace, workspaceSize)
end

@checked function cutensornetCreateSliceGroupFromIDs(handle, beginIDSequence, endIDSequence, sliceGroup)
    initialize_context()
    ccall((:cutensornetCreateSliceGroupFromIDs, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, Ptr{Int64}, Ptr{Int64}, Ptr{cutensornetSliceGroup_t}), handle, beginIDSequence, endIDSequence, sliceGroup)
end

@checked function cutensornetSetDeviceMemHandler(handle, devMemHandler)
    initialize_context()
    ccall((:cutensornetSetDeviceMemHandler, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, Ptr{cutensornetDeviceMemHandler_t}), handle, devMemHandler)
end

@checked function cutensornetCreateContractionOptimizerInfoFromPackedData(handle, descNet, buffer, sizeInBytes, optimizerInfo)
    initialize_context()
    ccall((:cutensornetCreateContractionOptimizerInfoFromPackedData, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetNetworkDescriptor_t, Ptr{Cvoid}, Csize_t, Ptr{cutensornetContractionOptimizerInfo_t}), handle, descNet, buffer, sizeInBytes, optimizerInfo)
end

@checked function cutensornetContractionAutotunePreferenceSetAttribute(handle, autotunePreference, attr, buf, sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionAutotunePreferenceSetAttribute, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionAutotunePreference_t, cutensornetContractionAutotunePreferenceAttributes_t, Ptr{Cvoid}, Csize_t), handle, autotunePreference, attr, buf, sizeInBytes)
end

@checked function cutensornetContractionAutotune(handle, plan, rawDataIn, rawDataOut, workDesc, pref, stream)
    initialize_context()
    ccall((:cutensornetContractionAutotune, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionPlan_t, Ptr{CuPtr{Cvoid}}, CuPtr{Cvoid}, cutensornetWorkspaceDescriptor_t, cutensornetContractionAutotunePreference_t, CUstream), handle, plan, rawDataIn, rawDataOut, workDesc, pref, stream)
end

@checked function cutensornetCreateContractionPlan(handle, descNet, optimizerInfo, workDesc, plan)
    initialize_context()
    ccall((:cutensornetCreateContractionPlan, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetNetworkDescriptor_t, cutensornetContractionOptimizerInfo_t, cutensornetWorkspaceDescriptor_t, Ptr{cutensornetContractionPlan_t}), handle, descNet, optimizerInfo, workDesc, plan)
end
@checked function cutensornetContraction(handle, plan, rawDataIn, rawDataOut, workDesc, sliceId, stream)
    initialize_context()
    ccall((:cutensornetContraction, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionPlan_t, Ptr{CuPtr{Cvoid}}, CuPtr{Cvoid}, cutensornetWorkspaceDescriptor_t, Int64, CUstream), handle, plan, rawDataIn, rawDataOut, workDesc, sliceId, stream)
end
