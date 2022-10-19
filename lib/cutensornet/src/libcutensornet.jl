
function cutensornetGetCudartVersion()
    ccall((:cutensornetGetCudartVersion, libcutensornet), Csize_t, ())
end

@checked function cutensornetContraction(handle, plan, rawDataIn, rawDataOut, workspace, workspaceSize, sliceId, stream)
    initialize_context()
    ccall((:cutensornetContraction, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionPlan_t, Ptr{CuPtr{Cvoid}}, CuPtr{Cvoid}, CuPtr{Cvoid}, UInt64, Int64, CUstream), handle, plan, rawDataIn, rawDataOut, workspace, workspaceSize, sliceId, stream)
end

@checked function cutensornetCreateContractionOptimizerConfig(handle, optimizerConfig)
    initialize_context()
    ccall((:cutensornetCreateContractionOptimizerConfig, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, Ptr{cutensornetContractionOptimizerConfig_t}), handle, optimizerConfig)
end

@checked function cutensornetCreate(handle)
    initialize_context()
    ccall((:cutensornetCreate, libcutensornet), cutensornetStatus_t, (Ptr{cutensornetHandle_t},), handle)
end

@checked function cutensornetContractionGetWorkspaceSize(handle, descNet, optimizerInfo, workspaceSize)
    initialize_context()
    ccall((:cutensornetContractionGetWorkspaceSize, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetNetworkDescriptor_t, cutensornetContractionOptimizerInfo_t, Ptr{UInt64}), handle, descNet, optimizerInfo, workspaceSize)
end

@checked function cutensornetDestroyContractionOptimizerInfo(optimizerInfo)
    initialize_context()
    ccall((:cutensornetDestroyContractionOptimizerInfo, libcutensornet), cutensornetStatus_t, (cutensornetContractionOptimizerInfo_t,), optimizerInfo)
end

@checked function cutensornetLoggerOpenFile(logFile)
    ccall((:cutensornetLoggerOpenFile, libcutensornet), cutensornetStatus_t, (Cstring,), logFile)
end

@checked function cutensornetContractionAutotunePreferenceSetAttribute(handle, autotunePreference, attr, buf, sizeInBytes)
    initialize_context()
    ccall((:cutensornetContractionAutotunePreferenceSetAttribute, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionAutotunePreference_t, cutensornetContractionAutotunePreferenceAttributes_t, Ptr{Cvoid}, Csize_t), handle, autotunePreference, attr, buf, sizeInBytes)
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

@checked function cutensornetContractionAutotune(handle, plan, rawDataIn, rawDataOut, workspace, workspaceSize, pref, stream)
    initialize_context()
    ccall((:cutensornetContractionAutotune, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetContractionPlan_t, Ptr{CuPtr{Cvoid}}, CuPtr{Cvoid}, CuPtr{Cvoid}, UInt64, cutensornetContractionAutotunePreference_t, CUstream), handle, plan, rawDataIn, rawDataOut, workspace, workspaceSize, pref, stream)
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

@checked function cutensornetCreateContractionPlan(handle, descNet, optimizerInfo, workspaceSize, plan)
    initialize_context()
    ccall((:cutensornetCreateContractionPlan, libcutensornet), cutensornetStatus_t, (cutensornetHandle_t, cutensornetNetworkDescriptor_t, cutensornetContractionOptimizerInfo_t, UInt64, Ptr{cutensornetContractionPlan_t}), handle, descNet, optimizerInfo, workspaceSize, plan)
end
