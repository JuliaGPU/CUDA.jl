# Julia wrapper for header: nvml.h
# Automatically generated using Clang.jl

@checked function nvmlInit_v2()
    ccall((:nvmlInit_v2, libnvml()), nvmlReturn_t, ())
end

@checked function nvmlInitWithFlags(flags)
    ccall((:nvmlInitWithFlags, libnvml()), nvmlReturn_t,
                   (UInt32,),
                   flags)
end

@checked function nvmlShutdown()
    ccall((:nvmlShutdown, libnvml()), nvmlReturn_t, ())
end

function nvmlErrorString(result)
    ccall((:nvmlErrorString, libnvml()), Cstring,
                   (nvmlReturn_t,),
                   result)
end

@checked function nvmlSystemGetDriverVersion(version, length)
    initialize_context()
    ccall((:nvmlSystemGetDriverVersion, libnvml()), nvmlReturn_t,
                   (Cstring, UInt32),
                   version, length)
end

@checked function nvmlSystemGetNVMLVersion(version, length)
    initialize_context()
    ccall((:nvmlSystemGetNVMLVersion, libnvml()), nvmlReturn_t,
                   (Cstring, UInt32),
                   version, length)
end

@checked function nvmlSystemGetCudaDriverVersion(cudaDriverVersion)
    initialize_context()
    ccall((:nvmlSystemGetCudaDriverVersion, libnvml()), nvmlReturn_t,
                   (Ptr{Cint},),
                   cudaDriverVersion)
end

@checked function nvmlSystemGetCudaDriverVersion_v2(cudaDriverVersion)
    initialize_context()
    ccall((:nvmlSystemGetCudaDriverVersion_v2, libnvml()), nvmlReturn_t,
                   (Ptr{Cint},),
                   cudaDriverVersion)
end

@checked function nvmlSystemGetProcessName(pid, name, length)
    initialize_context()
    ccall((:nvmlSystemGetProcessName, libnvml()), nvmlReturn_t,
                   (UInt32, Cstring, UInt32),
                   pid, name, length)
end

@checked function nvmlUnitGetCount(unitCount)
    initialize_context()
    ccall((:nvmlUnitGetCount, libnvml()), nvmlReturn_t,
                   (Ptr{UInt32},),
                   unitCount)
end

@checked function nvmlUnitGetHandleByIndex(index, unit)
    initialize_context()
    ccall((:nvmlUnitGetHandleByIndex, libnvml()), nvmlReturn_t,
                   (UInt32, Ptr{nvmlUnit_t}),
                   index, unit)
end

@checked function nvmlUnitGetUnitInfo(unit, info)
    initialize_context()
    ccall((:nvmlUnitGetUnitInfo, libnvml()), nvmlReturn_t,
                   (nvmlUnit_t, Ptr{nvmlUnitInfo_t}),
                   unit, info)
end

@checked function nvmlUnitGetLedState(unit, state)
    initialize_context()
    ccall((:nvmlUnitGetLedState, libnvml()), nvmlReturn_t,
                   (nvmlUnit_t, Ptr{nvmlLedState_t}),
                   unit, state)
end

@checked function nvmlUnitGetPsuInfo(unit, psu)
    initialize_context()
    ccall((:nvmlUnitGetPsuInfo, libnvml()), nvmlReturn_t,
                   (nvmlUnit_t, Ptr{nvmlPSUInfo_t}),
                   unit, psu)
end

@checked function nvmlUnitGetTemperature(unit, type, temp)
    initialize_context()
    ccall((:nvmlUnitGetTemperature, libnvml()), nvmlReturn_t,
                   (nvmlUnit_t, UInt32, Ptr{UInt32}),
                   unit, type, temp)
end

@checked function nvmlUnitGetFanSpeedInfo(unit, fanSpeeds)
    initialize_context()
    ccall((:nvmlUnitGetFanSpeedInfo, libnvml()), nvmlReturn_t,
                   (nvmlUnit_t, Ptr{nvmlUnitFanSpeeds_t}),
                   unit, fanSpeeds)
end

@checked function nvmlUnitGetDevices(unit, deviceCount, devices)
    initialize_context()
    ccall((:nvmlUnitGetDevices, libnvml()), nvmlReturn_t,
                   (nvmlUnit_t, Ptr{UInt32}, Ptr{nvmlDevice_t}),
                   unit, deviceCount, devices)
end

@checked function nvmlSystemGetHicVersion(hwbcCount, hwbcEntries)
    initialize_context()
    ccall((:nvmlSystemGetHicVersion, libnvml()), nvmlReturn_t,
                   (Ptr{UInt32}, Ptr{nvmlHwbcEntry_t}),
                   hwbcCount, hwbcEntries)
end

@checked function nvmlDeviceGetCount_v2(deviceCount)
    initialize_context()
    ccall((:nvmlDeviceGetCount_v2, libnvml()), nvmlReturn_t,
                   (Ptr{UInt32},),
                   deviceCount)
end

@checked function nvmlDeviceGetHandleByIndex_v2(index, device)
    initialize_context()
    ccall((:nvmlDeviceGetHandleByIndex_v2, libnvml()), nvmlReturn_t,
                   (UInt32, Ptr{nvmlDevice_t}),
                   index, device)
end

@checked function nvmlDeviceGetHandleBySerial(serial, device)
    initialize_context()
    ccall((:nvmlDeviceGetHandleBySerial, libnvml()), nvmlReturn_t,
                   (Cstring, Ptr{nvmlDevice_t}),
                   serial, device)
end

@checked function nvmlDeviceGetHandleByUUID(uuid, device)
    initialize_context()
    ccall((:nvmlDeviceGetHandleByUUID, libnvml()), nvmlReturn_t,
                   (Cstring, Ptr{nvmlDevice_t}),
                   uuid, device)
end

@checked function nvmlDeviceGetHandleByPciBusId_v2(pciBusId, device)
    initialize_context()
    ccall((:nvmlDeviceGetHandleByPciBusId_v2, libnvml()), nvmlReturn_t,
                   (Cstring, Ptr{nvmlDevice_t}),
                   pciBusId, device)
end

@checked function nvmlDeviceGetName(device, name, length)
    initialize_context()
    ccall((:nvmlDeviceGetName, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Cstring, UInt32),
                   device, name, length)
end

@checked function nvmlDeviceGetBrand(device, type)
    initialize_context()
    ccall((:nvmlDeviceGetBrand, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlBrandType_t}),
                   device, type)
end

@checked function nvmlDeviceGetIndex(device, index)
    initialize_context()
    ccall((:nvmlDeviceGetIndex, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}),
                   device, index)
end

@checked function nvmlDeviceGetSerial(device, serial, length)
    initialize_context()
    ccall((:nvmlDeviceGetSerial, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Cstring, UInt32),
                   device, serial, length)
end

@checked function nvmlDeviceGetMemoryAffinity(device, nodeSetSize, nodeSet, scope)
    initialize_context()
    ccall((:nvmlDeviceGetMemoryAffinity, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, Ptr{Culong}, nvmlAffinityScope_t),
                   device, nodeSetSize, nodeSet, scope)
end

@checked function nvmlDeviceGetCpuAffinityWithinScope(device, cpuSetSize, cpuSet, scope)
    initialize_context()
    ccall((:nvmlDeviceGetCpuAffinityWithinScope, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, Ptr{Culong}, nvmlAffinityScope_t),
                   device, cpuSetSize, cpuSet, scope)
end

@checked function nvmlDeviceGetCpuAffinity(device, cpuSetSize, cpuSet)
    initialize_context()
    ccall((:nvmlDeviceGetCpuAffinity, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, Ptr{Culong}),
                   device, cpuSetSize, cpuSet)
end

@checked function nvmlDeviceSetCpuAffinity(device)
    initialize_context()
    ccall((:nvmlDeviceSetCpuAffinity, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t,),
                   device)
end

@checked function nvmlDeviceClearCpuAffinity(device)
    initialize_context()
    ccall((:nvmlDeviceClearCpuAffinity, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t,),
                   device)
end

@checked function nvmlDeviceGetTopologyCommonAncestor(device1, device2, pathInfo)
    initialize_context()
    ccall((:nvmlDeviceGetTopologyCommonAncestor, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlDevice_t, Ptr{nvmlGpuTopologyLevel_t}),
                   device1, device2, pathInfo)
end

@checked function nvmlDeviceGetTopologyNearestGpus(device, level, count, deviceArray)
    initialize_context()
    ccall((:nvmlDeviceGetTopologyNearestGpus, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlGpuTopologyLevel_t, Ptr{UInt32}, Ptr{nvmlDevice_t}),
                   device, level, count, deviceArray)
end

@checked function nvmlSystemGetTopologyGpuSet(cpuNumber, count, deviceArray)
    initialize_context()
    ccall((:nvmlSystemGetTopologyGpuSet, libnvml()), nvmlReturn_t,
                   (UInt32, Ptr{UInt32}, Ptr{nvmlDevice_t}),
                   cpuNumber, count, deviceArray)
end

@checked function nvmlDeviceGetP2PStatus(device1, device2, p2pIndex, p2pStatus)
    initialize_context()
    ccall((:nvmlDeviceGetP2PStatus, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlDevice_t, nvmlGpuP2PCapsIndex_t,
                    Ptr{nvmlGpuP2PStatus_t}),
                   device1, device2, p2pIndex, p2pStatus)
end

@checked function nvmlDeviceGetUUID(device, uuid, length)
    initialize_context()
    ccall((:nvmlDeviceGetUUID, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Cstring, UInt32),
                   device, uuid, length)
end

@checked function nvmlVgpuInstanceGetMdevUUID(vgpuInstance, mdevUuid, size)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetMdevUUID, libnvml()), nvmlReturn_t,
                   (nvmlVgpuInstance_t, Cstring, UInt32),
                   vgpuInstance, mdevUuid, size)
end

@checked function nvmlDeviceGetMinorNumber(device, minorNumber)
    initialize_context()
    ccall((:nvmlDeviceGetMinorNumber, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}),
                   device, minorNumber)
end

@checked function nvmlDeviceGetBoardPartNumber(device, partNumber, length)
    initialize_context()
    ccall((:nvmlDeviceGetBoardPartNumber, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Cstring, UInt32),
                   device, partNumber, length)
end

@checked function nvmlDeviceGetInforomVersion(device, object, version, length)
    initialize_context()
    ccall((:nvmlDeviceGetInforomVersion, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlInforomObject_t, Cstring, UInt32),
                   device, object, version, length)
end

@checked function nvmlDeviceGetInforomImageVersion(device, version, length)
    initialize_context()
    ccall((:nvmlDeviceGetInforomImageVersion, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Cstring, UInt32),
                   device, version, length)
end

@checked function nvmlDeviceGetInforomConfigurationChecksum(device, checksum)
    initialize_context()
    ccall((:nvmlDeviceGetInforomConfigurationChecksum, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}),
                   device, checksum)
end

@checked function nvmlDeviceValidateInforom(device)
    initialize_context()
    ccall((:nvmlDeviceValidateInforom, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t,),
                   device)
end

@checked function nvmlDeviceGetDisplayMode(device, display)
    initialize_context()
    ccall((:nvmlDeviceGetDisplayMode, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlEnableState_t}),
                   device, display)
end

@checked function nvmlDeviceGetDisplayActive(device, isActive)
    initialize_context()
    ccall((:nvmlDeviceGetDisplayActive, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlEnableState_t}),
                   device, isActive)
end

@checked function nvmlDeviceGetPersistenceMode(device, mode)
    initialize_context()
    ccall((:nvmlDeviceGetPersistenceMode, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlEnableState_t}),
                   device, mode)
end

@checked function nvmlDeviceGetPciInfo_v3(device, pci)
    initialize_context()
    ccall((:nvmlDeviceGetPciInfo_v3, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlPciInfo_t}),
                   device, pci)
end

@checked function nvmlDeviceGetMaxPcieLinkGeneration(device, maxLinkGen)
    initialize_context()
    ccall((:nvmlDeviceGetMaxPcieLinkGeneration, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}),
                   device, maxLinkGen)
end

@checked function nvmlDeviceGetMaxPcieLinkWidth(device, maxLinkWidth)
    initialize_context()
    ccall((:nvmlDeviceGetMaxPcieLinkWidth, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}),
                   device, maxLinkWidth)
end

@checked function nvmlDeviceGetCurrPcieLinkGeneration(device, currLinkGen)
    initialize_context()
    ccall((:nvmlDeviceGetCurrPcieLinkGeneration, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}),
                   device, currLinkGen)
end

@checked function nvmlDeviceGetCurrPcieLinkWidth(device, currLinkWidth)
    initialize_context()
    ccall((:nvmlDeviceGetCurrPcieLinkWidth, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}),
                   device, currLinkWidth)
end

@checked function nvmlDeviceGetPcieThroughput(device, counter, value)
    initialize_context()
    ccall((:nvmlDeviceGetPcieThroughput, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlPcieUtilCounter_t, Ptr{UInt32}),
                   device, counter, value)
end

@checked function nvmlDeviceGetPcieReplayCounter(device, value)
    initialize_context()
    ccall((:nvmlDeviceGetPcieReplayCounter, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}),
                   device, value)
end

@checked function nvmlDeviceGetClockInfo(device, type, clock)
    initialize_context()
    ccall((:nvmlDeviceGetClockInfo, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlClockType_t, Ptr{UInt32}),
                   device, type, clock)
end

@checked function nvmlDeviceGetMaxClockInfo(device, type, clock)
    initialize_context()
    ccall((:nvmlDeviceGetMaxClockInfo, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlClockType_t, Ptr{UInt32}),
                   device, type, clock)
end

@checked function nvmlDeviceGetApplicationsClock(device, clockType, clockMHz)
    initialize_context()
    ccall((:nvmlDeviceGetApplicationsClock, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlClockType_t, Ptr{UInt32}),
                   device, clockType, clockMHz)
end

@checked function nvmlDeviceGetDefaultApplicationsClock(device, clockType, clockMHz)
    initialize_context()
    ccall((:nvmlDeviceGetDefaultApplicationsClock, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlClockType_t, Ptr{UInt32}),
                   device, clockType, clockMHz)
end

@checked function nvmlDeviceResetApplicationsClocks(device)
    initialize_context()
    ccall((:nvmlDeviceResetApplicationsClocks, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t,),
                   device)
end

@checked function nvmlDeviceGetClock(device, clockType, clockId, clockMHz)
    initialize_context()
    ccall((:nvmlDeviceGetClock, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlClockType_t, nvmlClockId_t, Ptr{UInt32}),
                   device, clockType, clockId, clockMHz)
end

@checked function nvmlDeviceGetMaxCustomerBoostClock(device, clockType, clockMHz)
    initialize_context()
    ccall((:nvmlDeviceGetMaxCustomerBoostClock, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlClockType_t, Ptr{UInt32}),
                   device, clockType, clockMHz)
end

@checked function nvmlDeviceGetSupportedMemoryClocks(device, count, clocksMHz)
    initialize_context()
    ccall((:nvmlDeviceGetSupportedMemoryClocks, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}, Ptr{UInt32}),
                   device, count, clocksMHz)
end

@checked function nvmlDeviceGetSupportedGraphicsClocks(device, memoryClockMHz, count,
                                                       clocksMHz)
    initialize_context()
    ccall((:nvmlDeviceGetSupportedGraphicsClocks, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, Ptr{UInt32}, Ptr{UInt32}),
                   device, memoryClockMHz, count, clocksMHz)
end

@checked function nvmlDeviceGetAutoBoostedClocksEnabled(device, isEnabled, defaultIsEnabled)
    initialize_context()
    ccall((:nvmlDeviceGetAutoBoostedClocksEnabled, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlEnableState_t}, Ptr{nvmlEnableState_t}),
                   device, isEnabled, defaultIsEnabled)
end

@checked function nvmlDeviceSetAutoBoostedClocksEnabled(device, enabled)
    initialize_context()
    ccall((:nvmlDeviceSetAutoBoostedClocksEnabled, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlEnableState_t),
                   device, enabled)
end

@checked function nvmlDeviceSetDefaultAutoBoostedClocksEnabled(device, enabled, flags)
    initialize_context()
    ccall((:nvmlDeviceSetDefaultAutoBoostedClocksEnabled, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlEnableState_t, UInt32),
                   device, enabled, flags)
end

@checked function nvmlDeviceGetFanSpeed(device, speed)
    initialize_context()
    ccall((:nvmlDeviceGetFanSpeed, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}),
                   device, speed)
end

@checked function nvmlDeviceGetFanSpeed_v2(device, fan, speed)
    initialize_context()
    ccall((:nvmlDeviceGetFanSpeed_v2, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, Ptr{UInt32}),
                   device, fan, speed)
end

@checked function nvmlDeviceGetTemperature(device, sensorType, temp)
    initialize_context()
    ccall((:nvmlDeviceGetTemperature, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlTemperatureSensors_t, Ptr{UInt32}),
                   device, sensorType, temp)
end

@checked function nvmlDeviceGetTemperatureThreshold(device, thresholdType, temp)
    initialize_context()
    ccall((:nvmlDeviceGetTemperatureThreshold, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlTemperatureThresholds_t, Ptr{UInt32}),
                   device, thresholdType, temp)
end

@checked function nvmlDeviceGetPerformanceState(device, pState)
    initialize_context()
    ccall((:nvmlDeviceGetPerformanceState, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlPstates_t}),
                   device, pState)
end

@checked function nvmlDeviceGetCurrentClocksThrottleReasons(device, clocksThrottleReasons)
    initialize_context()
    ccall((:nvmlDeviceGetCurrentClocksThrottleReasons, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{Culonglong}),
                   device, clocksThrottleReasons)
end

@checked function nvmlDeviceGetSupportedClocksThrottleReasons(device,
                                                              supportedClocksThrottleReasons)
    initialize_context()
    ccall((:nvmlDeviceGetSupportedClocksThrottleReasons, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{Culonglong}),
                   device, supportedClocksThrottleReasons)
end

@checked function nvmlDeviceGetPowerState(device, pState)
    initialize_context()
    ccall((:nvmlDeviceGetPowerState, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlPstates_t}),
                   device, pState)
end

@checked function nvmlDeviceGetPowerManagementMode(device, mode)
    initialize_context()
    ccall((:nvmlDeviceGetPowerManagementMode, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlEnableState_t}),
                   device, mode)
end

@checked function nvmlDeviceGetPowerManagementLimit(device, limit)
    initialize_context()
    ccall((:nvmlDeviceGetPowerManagementLimit, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}),
                   device, limit)
end

@checked function nvmlDeviceGetPowerManagementLimitConstraints(device, minLimit, maxLimit)
    initialize_context()
    ccall((:nvmlDeviceGetPowerManagementLimitConstraints, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}, Ptr{UInt32}),
                   device, minLimit, maxLimit)
end

@checked function nvmlDeviceGetPowerManagementDefaultLimit(device, defaultLimit)
    initialize_context()
    ccall((:nvmlDeviceGetPowerManagementDefaultLimit, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}),
                   device, defaultLimit)
end

@checked function nvmlDeviceGetPowerUsage(device, power)
    initialize_context()
    ccall((:nvmlDeviceGetPowerUsage, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}),
                   device, power)
end

@checked function nvmlDeviceGetTotalEnergyConsumption(device, energy)
    initialize_context()
    ccall((:nvmlDeviceGetTotalEnergyConsumption, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{Culonglong}),
                   device, energy)
end

@checked function nvmlDeviceGetEnforcedPowerLimit(device, limit)
    initialize_context()
    ccall((:nvmlDeviceGetEnforcedPowerLimit, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}),
                   device, limit)
end

@checked function nvmlDeviceGetGpuOperationMode(device, current, pending)
    initialize_context()
    ccall((:nvmlDeviceGetGpuOperationMode, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlGpuOperationMode_t}, Ptr{nvmlGpuOperationMode_t}),
                   device, current, pending)
end

@checked function nvmlDeviceGetMemoryInfo(device, memory)
    initialize_context()
    ccall((:nvmlDeviceGetMemoryInfo, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlMemory_t}),
                   device, memory)
end

@checked function nvmlDeviceGetComputeMode(device, mode)
    initialize_context()
    ccall((:nvmlDeviceGetComputeMode, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlComputeMode_t}),
                   device, mode)
end

@checked function nvmlDeviceGetCudaComputeCapability(device, major, minor)
    initialize_context()
    ccall((:nvmlDeviceGetCudaComputeCapability, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{Cint}, Ptr{Cint}),
                   device, major, minor)
end

@checked function nvmlDeviceGetEccMode(device, current, pending)
    initialize_context()
    ccall((:nvmlDeviceGetEccMode, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlEnableState_t}, Ptr{nvmlEnableState_t}),
                   device, current, pending)
end

@checked function nvmlDeviceGetBoardId(device, boardId)
    initialize_context()
    ccall((:nvmlDeviceGetBoardId, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}),
                   device, boardId)
end

@checked function nvmlDeviceGetMultiGpuBoard(device, multiGpuBool)
    initialize_context()
    ccall((:nvmlDeviceGetMultiGpuBoard, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}),
                   device, multiGpuBool)
end

@checked function nvmlDeviceGetTotalEccErrors(device, errorType, counterType, eccCounts)
    initialize_context()
    ccall((:nvmlDeviceGetTotalEccErrors, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlMemoryErrorType_t, nvmlEccCounterType_t,
                    Ptr{Culonglong}),
                   device, errorType, counterType, eccCounts)
end

@checked function nvmlDeviceGetDetailedEccErrors(device, errorType, counterType, eccCounts)
    initialize_context()
    ccall((:nvmlDeviceGetDetailedEccErrors, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlMemoryErrorType_t, nvmlEccCounterType_t,
                    Ptr{nvmlEccErrorCounts_t}),
                   device, errorType, counterType, eccCounts)
end

@checked function nvmlDeviceGetMemoryErrorCounter(device, errorType, counterType,
                                                  locationType, count)
    initialize_context()
    ccall((:nvmlDeviceGetMemoryErrorCounter, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlMemoryErrorType_t, nvmlEccCounterType_t,
                    nvmlMemoryLocation_t, Ptr{Culonglong}),
                   device, errorType, counterType, locationType, count)
end

@checked function nvmlDeviceGetUtilizationRates(device, utilization)
    initialize_context()
    ccall((:nvmlDeviceGetUtilizationRates, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlUtilization_t}),
                   device, utilization)
end

@checked function nvmlDeviceGetEncoderUtilization(device, utilization, samplingPeriodUs)
    initialize_context()
    ccall((:nvmlDeviceGetEncoderUtilization, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}, Ptr{UInt32}),
                   device, utilization, samplingPeriodUs)
end

@checked function nvmlDeviceGetEncoderCapacity(device, encoderQueryType, encoderCapacity)
    initialize_context()
    ccall((:nvmlDeviceGetEncoderCapacity, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlEncoderType_t, Ptr{UInt32}),
                   device, encoderQueryType, encoderCapacity)
end

@checked function nvmlDeviceGetEncoderStats(device, sessionCount, averageFps, averageLatency)
    initialize_context()
    ccall((:nvmlDeviceGetEncoderStats, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}, Ptr{UInt32}, Ptr{UInt32}),
                   device, sessionCount, averageFps, averageLatency)
end

@checked function nvmlDeviceGetEncoderSessions(device, sessionCount, sessionInfos)
    initialize_context()
    ccall((:nvmlDeviceGetEncoderSessions, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}, Ptr{nvmlEncoderSessionInfo_t}),
                   device, sessionCount, sessionInfos)
end

@checked function nvmlDeviceGetDecoderUtilization(device, utilization, samplingPeriodUs)
    initialize_context()
    ccall((:nvmlDeviceGetDecoderUtilization, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}, Ptr{UInt32}),
                   device, utilization, samplingPeriodUs)
end

@checked function nvmlDeviceGetFBCStats(device, fbcStats)
    initialize_context()
    ccall((:nvmlDeviceGetFBCStats, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlFBCStats_t}),
                   device, fbcStats)
end

@checked function nvmlDeviceGetFBCSessions(device, sessionCount, sessionInfo)
    initialize_context()
    ccall((:nvmlDeviceGetFBCSessions, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}, Ptr{nvmlFBCSessionInfo_t}),
                   device, sessionCount, sessionInfo)
end

@checked function nvmlDeviceGetDriverModel(device, current, pending)
    initialize_context()
    ccall((:nvmlDeviceGetDriverModel, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlDriverModel_t}, Ptr{nvmlDriverModel_t}),
                   device, current, pending)
end

@checked function nvmlDeviceGetVbiosVersion(device, version, length)
    initialize_context()
    ccall((:nvmlDeviceGetVbiosVersion, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Cstring, UInt32),
                   device, version, length)
end

@checked function nvmlDeviceGetBridgeChipInfo(device, bridgeHierarchy)
    initialize_context()
    ccall((:nvmlDeviceGetBridgeChipInfo, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlBridgeChipHierarchy_t}),
                   device, bridgeHierarchy)
end

@checked function nvmlDeviceOnSameBoard(device1, device2, onSameBoard)
    initialize_context()
    ccall((:nvmlDeviceOnSameBoard, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlDevice_t, Ptr{Cint}),
                   device1, device2, onSameBoard)
end

@checked function nvmlDeviceGetAPIRestriction(device, apiType, isRestricted)
    initialize_context()
    ccall((:nvmlDeviceGetAPIRestriction, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlRestrictedAPI_t, Ptr{nvmlEnableState_t}),
                   device, apiType, isRestricted)
end

@checked function nvmlDeviceGetSamples(device, type, lastSeenTimeStamp, sampleValType,
                                       sampleCount, samples)
    initialize_context()
    ccall((:nvmlDeviceGetSamples, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlSamplingType_t, Culonglong, Ptr{nvmlValueType_t},
                    Ptr{UInt32}, Ptr{nvmlSample_t}),
                   device, type, lastSeenTimeStamp, sampleValType, sampleCount, samples)
end

@checked function nvmlDeviceGetBAR1MemoryInfo(device, bar1Memory)
    initialize_context()
    ccall((:nvmlDeviceGetBAR1MemoryInfo, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlBAR1Memory_t}),
                   device, bar1Memory)
end

@checked function nvmlDeviceGetViolationStatus(device, perfPolicyType, violTime)
    initialize_context()
    ccall((:nvmlDeviceGetViolationStatus, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlPerfPolicyType_t, Ptr{nvmlViolationTime_t}),
                   device, perfPolicyType, violTime)
end

@checked function nvmlDeviceGetAccountingMode(device, mode)
    initialize_context()
    ccall((:nvmlDeviceGetAccountingMode, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlEnableState_t}),
                   device, mode)
end

@checked function nvmlDeviceGetAccountingStats(device, pid, stats)
    initialize_context()
    ccall((:nvmlDeviceGetAccountingStats, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, Ptr{nvmlAccountingStats_t}),
                   device, pid, stats)
end

@checked function nvmlDeviceGetAccountingPids(device, count, pids)
    initialize_context()
    ccall((:nvmlDeviceGetAccountingPids, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}, Ptr{UInt32}),
                   device, count, pids)
end

@checked function nvmlDeviceGetAccountingBufferSize(device, bufferSize)
    initialize_context()
    ccall((:nvmlDeviceGetAccountingBufferSize, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}),
                   device, bufferSize)
end

@checked function nvmlDeviceGetRetiredPages(device, cause, pageCount, addresses)
    initialize_context()
    ccall((:nvmlDeviceGetRetiredPages, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlPageRetirementCause_t, Ptr{UInt32}, Ptr{Culonglong}),
                   device, cause, pageCount, addresses)
end

@checked function nvmlDeviceGetRetiredPages_v2(device, cause, pageCount, addresses,
                                               timestamps)
    initialize_context()
    ccall((:nvmlDeviceGetRetiredPages_v2, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlPageRetirementCause_t, Ptr{UInt32}, Ptr{Culonglong},
                    Ptr{Culonglong}),
                   device, cause, pageCount, addresses, timestamps)
end

@checked function nvmlDeviceGetRetiredPagesPendingStatus(device, isPending)
    initialize_context()
    ccall((:nvmlDeviceGetRetiredPagesPendingStatus, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlEnableState_t}),
                   device, isPending)
end

@checked function nvmlDeviceGetRemappedRows(device, corrRows, uncRows, isPending,
                                            failureOccurred)
    initialize_context()
    ccall((:nvmlDeviceGetRemappedRows, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}, Ptr{UInt32}, Ptr{UInt32}, Ptr{UInt32}),
                   device, corrRows, uncRows, isPending, failureOccurred)
end

@checked function nvmlDeviceGetArchitecture(device, arch)
    initialize_context()
    ccall((:nvmlDeviceGetArchitecture, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlDeviceArchitecture_t}),
                   device, arch)
end

@checked function nvmlUnitSetLedState(unit, color)
    initialize_context()
    ccall((:nvmlUnitSetLedState, libnvml()), nvmlReturn_t,
                   (nvmlUnit_t, nvmlLedColor_t),
                   unit, color)
end

@checked function nvmlDeviceSetPersistenceMode(device, mode)
    initialize_context()
    ccall((:nvmlDeviceSetPersistenceMode, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlEnableState_t),
                   device, mode)
end

@checked function nvmlDeviceSetComputeMode(device, mode)
    initialize_context()
    ccall((:nvmlDeviceSetComputeMode, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlComputeMode_t),
                   device, mode)
end

@checked function nvmlDeviceSetEccMode(device, ecc)
    initialize_context()
    ccall((:nvmlDeviceSetEccMode, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlEnableState_t),
                   device, ecc)
end

@checked function nvmlDeviceClearEccErrorCounts(device, counterType)
    initialize_context()
    ccall((:nvmlDeviceClearEccErrorCounts, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlEccCounterType_t),
                   device, counterType)
end

@checked function nvmlDeviceSetDriverModel(device, driverModel, flags)
    initialize_context()
    ccall((:nvmlDeviceSetDriverModel, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlDriverModel_t, UInt32),
                   device, driverModel, flags)
end

@checked function nvmlDeviceSetGpuLockedClocks(device, minGpuClockMHz, maxGpuClockMHz)
    initialize_context()
    ccall((:nvmlDeviceSetGpuLockedClocks, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, UInt32),
                   device, minGpuClockMHz, maxGpuClockMHz)
end

@checked function nvmlDeviceResetGpuLockedClocks(device)
    initialize_context()
    ccall((:nvmlDeviceResetGpuLockedClocks, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t,),
                   device)
end

@checked function nvmlDeviceSetApplicationsClocks(device, memClockMHz, graphicsClockMHz)
    initialize_context()
    ccall((:nvmlDeviceSetApplicationsClocks, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, UInt32),
                   device, memClockMHz, graphicsClockMHz)
end

@checked function nvmlDeviceSetPowerManagementLimit(device, limit)
    initialize_context()
    ccall((:nvmlDeviceSetPowerManagementLimit, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32),
                   device, limit)
end

@checked function nvmlDeviceSetGpuOperationMode(device, mode)
    initialize_context()
    ccall((:nvmlDeviceSetGpuOperationMode, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlGpuOperationMode_t),
                   device, mode)
end

@checked function nvmlDeviceSetAPIRestriction(device, apiType, isRestricted)
    initialize_context()
    ccall((:nvmlDeviceSetAPIRestriction, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlRestrictedAPI_t, nvmlEnableState_t),
                   device, apiType, isRestricted)
end

@checked function nvmlDeviceSetAccountingMode(device, mode)
    initialize_context()
    ccall((:nvmlDeviceSetAccountingMode, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlEnableState_t),
                   device, mode)
end

@checked function nvmlDeviceClearAccountingPids(device)
    initialize_context()
    ccall((:nvmlDeviceClearAccountingPids, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t,),
                   device)
end

@checked function nvmlDeviceGetNvLinkState(device, link, isActive)
    initialize_context()
    ccall((:nvmlDeviceGetNvLinkState, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, Ptr{nvmlEnableState_t}),
                   device, link, isActive)
end

@checked function nvmlDeviceGetNvLinkVersion(device, link, version)
    initialize_context()
    ccall((:nvmlDeviceGetNvLinkVersion, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, Ptr{UInt32}),
                   device, link, version)
end

@checked function nvmlDeviceGetNvLinkCapability(device, link, capability, capResult)
    initialize_context()
    ccall((:nvmlDeviceGetNvLinkCapability, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, nvmlNvLinkCapability_t, Ptr{UInt32}),
                   device, link, capability, capResult)
end

@checked function nvmlDeviceGetNvLinkRemotePciInfo_v2(device, link, pci)
    initialize_context()
    ccall((:nvmlDeviceGetNvLinkRemotePciInfo_v2, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, Ptr{nvmlPciInfo_t}),
                   device, link, pci)
end

@checked function nvmlDeviceGetNvLinkErrorCounter(device, link, counter, counterValue)
    initialize_context()
    ccall((:nvmlDeviceGetNvLinkErrorCounter, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, nvmlNvLinkErrorCounter_t, Ptr{Culonglong}),
                   device, link, counter, counterValue)
end

@checked function nvmlDeviceResetNvLinkErrorCounters(device, link)
    initialize_context()
    ccall((:nvmlDeviceResetNvLinkErrorCounters, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32),
                   device, link)
end

@checked function nvmlDeviceSetNvLinkUtilizationControl(device, link, counter, control,
                                                        reset)
    initialize_context()
    ccall((:nvmlDeviceSetNvLinkUtilizationControl, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, UInt32, Ptr{nvmlNvLinkUtilizationControl_t},
                    UInt32),
                   device, link, counter, control, reset)
end

@checked function nvmlDeviceGetNvLinkUtilizationControl(device, link, counter, control)
    initialize_context()
    ccall((:nvmlDeviceGetNvLinkUtilizationControl, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, UInt32, Ptr{nvmlNvLinkUtilizationControl_t}),
                   device, link, counter, control)
end

@checked function nvmlDeviceGetNvLinkUtilizationCounter(device, link, counter, rxcounter,
                                                        txcounter)
    initialize_context()
    ccall((:nvmlDeviceGetNvLinkUtilizationCounter, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, UInt32, Ptr{Culonglong}, Ptr{Culonglong}),
                   device, link, counter, rxcounter, txcounter)
end

@checked function nvmlDeviceFreezeNvLinkUtilizationCounter(device, link, counter, freeze)
    initialize_context()
    ccall((:nvmlDeviceFreezeNvLinkUtilizationCounter, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, UInt32, nvmlEnableState_t),
                   device, link, counter, freeze)
end

@checked function nvmlDeviceResetNvLinkUtilizationCounter(device, link, counter)
    initialize_context()
    ccall((:nvmlDeviceResetNvLinkUtilizationCounter, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, UInt32),
                   device, link, counter)
end

@checked function nvmlEventSetCreate(set)
    initialize_context()
    ccall((:nvmlEventSetCreate, libnvml()), nvmlReturn_t,
                   (Ptr{nvmlEventSet_t},),
                   set)
end

@checked function nvmlDeviceRegisterEvents(device, eventTypes, set)
    initialize_context()
    ccall((:nvmlDeviceRegisterEvents, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Culonglong, nvmlEventSet_t),
                   device, eventTypes, set)
end

@checked function nvmlDeviceGetSupportedEventTypes(device, eventTypes)
    initialize_context()
    ccall((:nvmlDeviceGetSupportedEventTypes, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{Culonglong}),
                   device, eventTypes)
end

@checked function nvmlEventSetWait_v2(set, data, timeoutms)
    initialize_context()
    ccall((:nvmlEventSetWait_v2, libnvml()), nvmlReturn_t,
                   (nvmlEventSet_t, Ptr{nvmlEventData_t}, UInt32),
                   set, data, timeoutms)
end

@checked function nvmlEventSetFree(set)
    initialize_context()
    ccall((:nvmlEventSetFree, libnvml()), nvmlReturn_t,
                   (nvmlEventSet_t,),
                   set)
end

@checked function nvmlDeviceModifyDrainState(pciInfo, newState)
    initialize_context()
    ccall((:nvmlDeviceModifyDrainState, libnvml()), nvmlReturn_t,
                   (Ptr{nvmlPciInfo_t}, nvmlEnableState_t),
                   pciInfo, newState)
end

@checked function nvmlDeviceQueryDrainState(pciInfo, currentState)
    initialize_context()
    ccall((:nvmlDeviceQueryDrainState, libnvml()), nvmlReturn_t,
                   (Ptr{nvmlPciInfo_t}, Ptr{nvmlEnableState_t}),
                   pciInfo, currentState)
end

@checked function nvmlDeviceRemoveGpu_v2(pciInfo, gpuState, linkState)
    initialize_context()
    ccall((:nvmlDeviceRemoveGpu_v2, libnvml()), nvmlReturn_t,
                   (Ptr{nvmlPciInfo_t}, nvmlDetachGpuState_t, nvmlPcieLinkState_t),
                   pciInfo, gpuState, linkState)
end

@checked function nvmlDeviceDiscoverGpus(pciInfo)
    initialize_context()
    ccall((:nvmlDeviceDiscoverGpus, libnvml()), nvmlReturn_t,
                   (Ptr{nvmlPciInfo_t},),
                   pciInfo)
end

@checked function nvmlDeviceGetFieldValues(device, valuesCount, values)
    initialize_context()
    ccall((:nvmlDeviceGetFieldValues, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Cint, Ptr{nvmlFieldValue_t}),
                   device, valuesCount, values)
end

@checked function nvmlDeviceGetVirtualizationMode(device, pVirtualMode)
    initialize_context()
    ccall((:nvmlDeviceGetVirtualizationMode, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlGpuVirtualizationMode_t}),
                   device, pVirtualMode)
end

@checked function nvmlDeviceGetHostVgpuMode(device, pHostVgpuMode)
    initialize_context()
    ccall((:nvmlDeviceGetHostVgpuMode, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlHostVgpuMode_t}),
                   device, pHostVgpuMode)
end

@checked function nvmlDeviceSetVirtualizationMode(device, virtualMode)
    initialize_context()
    ccall((:nvmlDeviceSetVirtualizationMode, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlGpuVirtualizationMode_t),
                   device, virtualMode)
end

@checked function nvmlDeviceGetGridLicensableFeatures_v3(device, pGridLicensableFeatures)
    initialize_context()
    ccall((:nvmlDeviceGetGridLicensableFeatures_v3, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlGridLicensableFeatures_t}),
                   device, pGridLicensableFeatures)
end

@checked function nvmlDeviceGetProcessUtilization(device, utilization, processSamplesCount,
                                                  lastSeenTimeStamp)
    initialize_context()
    ccall((:nvmlDeviceGetProcessUtilization, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlProcessUtilizationSample_t}, Ptr{UInt32},
                    Culonglong),
                   device, utilization, processSamplesCount, lastSeenTimeStamp)
end

@checked function nvmlDeviceGetSupportedVgpus(device, vgpuCount, vgpuTypeIds)
    initialize_context()
    ccall((:nvmlDeviceGetSupportedVgpus, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}, Ptr{nvmlVgpuTypeId_t}),
                   device, vgpuCount, vgpuTypeIds)
end

@checked function nvmlDeviceGetCreatableVgpus(device, vgpuCount, vgpuTypeIds)
    initialize_context()
    ccall((:nvmlDeviceGetCreatableVgpus, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}, Ptr{nvmlVgpuTypeId_t}),
                   device, vgpuCount, vgpuTypeIds)
end

@checked function nvmlVgpuTypeGetClass(vgpuTypeId, vgpuTypeClass, size)
    initialize_context()
    ccall((:nvmlVgpuTypeGetClass, libnvml()), nvmlReturn_t,
                   (nvmlVgpuTypeId_t, Cstring, Ptr{UInt32}),
                   vgpuTypeId, vgpuTypeClass, size)
end

@checked function nvmlVgpuTypeGetName(vgpuTypeId, vgpuTypeName, size)
    initialize_context()
    ccall((:nvmlVgpuTypeGetName, libnvml()), nvmlReturn_t,
                   (nvmlVgpuTypeId_t, Cstring, Ptr{UInt32}),
                   vgpuTypeId, vgpuTypeName, size)
end

@checked function nvmlVgpuTypeGetDeviceID(vgpuTypeId, deviceID, subsystemID)
    initialize_context()
    ccall((:nvmlVgpuTypeGetDeviceID, libnvml()), nvmlReturn_t,
                   (nvmlVgpuTypeId_t, Ptr{Culonglong}, Ptr{Culonglong}),
                   vgpuTypeId, deviceID, subsystemID)
end

@checked function nvmlVgpuTypeGetFramebufferSize(vgpuTypeId, fbSize)
    initialize_context()
    ccall((:nvmlVgpuTypeGetFramebufferSize, libnvml()), nvmlReturn_t,
                   (nvmlVgpuTypeId_t, Ptr{Culonglong}),
                   vgpuTypeId, fbSize)
end

@checked function nvmlVgpuTypeGetNumDisplayHeads(vgpuTypeId, numDisplayHeads)
    initialize_context()
    ccall((:nvmlVgpuTypeGetNumDisplayHeads, libnvml()), nvmlReturn_t,
                   (nvmlVgpuTypeId_t, Ptr{UInt32}),
                   vgpuTypeId, numDisplayHeads)
end

@checked function nvmlVgpuTypeGetResolution(vgpuTypeId, displayIndex, xdim, ydim)
    initialize_context()
    ccall((:nvmlVgpuTypeGetResolution, libnvml()), nvmlReturn_t,
                   (nvmlVgpuTypeId_t, UInt32, Ptr{UInt32}, Ptr{UInt32}),
                   vgpuTypeId, displayIndex, xdim, ydim)
end

@checked function nvmlVgpuTypeGetLicense(vgpuTypeId, vgpuTypeLicenseString, size)
    initialize_context()
    ccall((:nvmlVgpuTypeGetLicense, libnvml()), nvmlReturn_t,
                   (nvmlVgpuTypeId_t, Cstring, UInt32),
                   vgpuTypeId, vgpuTypeLicenseString, size)
end

@checked function nvmlVgpuTypeGetFrameRateLimit(vgpuTypeId, frameRateLimit)
    initialize_context()
    ccall((:nvmlVgpuTypeGetFrameRateLimit, libnvml()), nvmlReturn_t,
                   (nvmlVgpuTypeId_t, Ptr{UInt32}),
                   vgpuTypeId, frameRateLimit)
end

@checked function nvmlVgpuTypeGetMaxInstances(device, vgpuTypeId, vgpuInstanceCount)
    initialize_context()
    ccall((:nvmlVgpuTypeGetMaxInstances, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, nvmlVgpuTypeId_t, Ptr{UInt32}),
                   device, vgpuTypeId, vgpuInstanceCount)
end

@checked function nvmlVgpuTypeGetMaxInstancesPerVm(vgpuTypeId, vgpuInstanceCountPerVm)
    initialize_context()
    ccall((:nvmlVgpuTypeGetMaxInstancesPerVm, libnvml()), nvmlReturn_t,
                   (nvmlVgpuTypeId_t, Ptr{UInt32}),
                   vgpuTypeId, vgpuInstanceCountPerVm)
end

@checked function nvmlDeviceGetActiveVgpus(device, vgpuCount, vgpuInstances)
    initialize_context()
    ccall((:nvmlDeviceGetActiveVgpus, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}, Ptr{nvmlVgpuInstance_t}),
                   device, vgpuCount, vgpuInstances)
end

@checked function nvmlVgpuInstanceGetVmID(vgpuInstance, vmId, size, vmIdType)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetVmID, libnvml()), nvmlReturn_t,
                   (nvmlVgpuInstance_t, Cstring, UInt32, Ptr{nvmlVgpuVmIdType_t}),
                   vgpuInstance, vmId, size, vmIdType)
end

@checked function nvmlVgpuInstanceGetUUID(vgpuInstance, uuid, size)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetUUID, libnvml()), nvmlReturn_t,
                   (nvmlVgpuInstance_t, Cstring, UInt32),
                   vgpuInstance, uuid, size)
end

@checked function nvmlVgpuInstanceGetVmDriverVersion(vgpuInstance, version, length)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetVmDriverVersion, libnvml()), nvmlReturn_t,
                   (nvmlVgpuInstance_t, Cstring, UInt32),
                   vgpuInstance, version, length)
end

@checked function nvmlVgpuInstanceGetFbUsage(vgpuInstance, fbUsage)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetFbUsage, libnvml()), nvmlReturn_t,
                   (nvmlVgpuInstance_t, Ptr{Culonglong}),
                   vgpuInstance, fbUsage)
end

@checked function nvmlVgpuInstanceGetLicenseStatus(vgpuInstance, licensed)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetLicenseStatus, libnvml()), nvmlReturn_t,
                   (nvmlVgpuInstance_t, Ptr{UInt32}),
                   vgpuInstance, licensed)
end

@checked function nvmlVgpuInstanceGetType(vgpuInstance, vgpuTypeId)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetType, libnvml()), nvmlReturn_t,
                   (nvmlVgpuInstance_t, Ptr{nvmlVgpuTypeId_t}),
                   vgpuInstance, vgpuTypeId)
end

@checked function nvmlVgpuInstanceGetFrameRateLimit(vgpuInstance, frameRateLimit)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetFrameRateLimit, libnvml()), nvmlReturn_t,
                   (nvmlVgpuInstance_t, Ptr{UInt32}),
                   vgpuInstance, frameRateLimit)
end

@checked function nvmlVgpuInstanceGetEccMode(vgpuInstance, eccMode)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetEccMode, libnvml()), nvmlReturn_t,
                   (nvmlVgpuInstance_t, Ptr{nvmlEnableState_t}),
                   vgpuInstance, eccMode)
end

@checked function nvmlVgpuInstanceGetEncoderCapacity(vgpuInstance, encoderCapacity)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetEncoderCapacity, libnvml()), nvmlReturn_t,
                   (nvmlVgpuInstance_t, Ptr{UInt32}),
                   vgpuInstance, encoderCapacity)
end

@checked function nvmlVgpuInstanceSetEncoderCapacity(vgpuInstance, encoderCapacity)
    initialize_context()
    ccall((:nvmlVgpuInstanceSetEncoderCapacity, libnvml()), nvmlReturn_t,
                   (nvmlVgpuInstance_t, UInt32),
                   vgpuInstance, encoderCapacity)
end

@checked function nvmlVgpuInstanceGetEncoderStats(vgpuInstance, sessionCount, averageFps,
                                                  averageLatency)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetEncoderStats, libnvml()), nvmlReturn_t,
                   (nvmlVgpuInstance_t, Ptr{UInt32}, Ptr{UInt32}, Ptr{UInt32}),
                   vgpuInstance, sessionCount, averageFps, averageLatency)
end

@checked function nvmlVgpuInstanceGetEncoderSessions(vgpuInstance, sessionCount, sessionInfo)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetEncoderSessions, libnvml()), nvmlReturn_t,
                   (nvmlVgpuInstance_t, Ptr{UInt32}, Ptr{nvmlEncoderSessionInfo_t}),
                   vgpuInstance, sessionCount, sessionInfo)
end

@checked function nvmlVgpuInstanceGetFBCStats(vgpuInstance, fbcStats)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetFBCStats, libnvml()), nvmlReturn_t,
                   (nvmlVgpuInstance_t, Ptr{nvmlFBCStats_t}),
                   vgpuInstance, fbcStats)
end

@checked function nvmlVgpuInstanceGetFBCSessions(vgpuInstance, sessionCount, sessionInfo)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetFBCSessions, libnvml()), nvmlReturn_t,
                   (nvmlVgpuInstance_t, Ptr{UInt32}, Ptr{nvmlFBCSessionInfo_t}),
                   vgpuInstance, sessionCount, sessionInfo)
end

@checked function nvmlVgpuInstanceGetMetadata(vgpuInstance, vgpuMetadata, bufferSize)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetMetadata, libnvml()), nvmlReturn_t,
                   (nvmlVgpuInstance_t, Ptr{nvmlVgpuMetadata_t}, Ptr{UInt32}),
                   vgpuInstance, vgpuMetadata, bufferSize)
end

@checked function nvmlDeviceGetVgpuMetadata(device, pgpuMetadata, bufferSize)
    initialize_context()
    ccall((:nvmlDeviceGetVgpuMetadata, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlVgpuPgpuMetadata_t}, Ptr{UInt32}),
                   device, pgpuMetadata, bufferSize)
end

@checked function nvmlGetVgpuCompatibility(vgpuMetadata, pgpuMetadata, compatibilityInfo)
    initialize_context()
    ccall((:nvmlGetVgpuCompatibility, libnvml()), nvmlReturn_t,
                   (Ptr{nvmlVgpuMetadata_t}, Ptr{nvmlVgpuPgpuMetadata_t},
                    Ptr{nvmlVgpuPgpuCompatibility_t}),
                   vgpuMetadata, pgpuMetadata, compatibilityInfo)
end

@checked function nvmlDeviceGetPgpuMetadataString(device, pgpuMetadata, bufferSize)
    initialize_context()
    ccall((:nvmlDeviceGetPgpuMetadataString, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Cstring, Ptr{UInt32}),
                   device, pgpuMetadata, bufferSize)
end

@checked function nvmlGetVgpuVersion(supported, current)
    initialize_context()
    ccall((:nvmlGetVgpuVersion, libnvml()), nvmlReturn_t,
                   (Ptr{nvmlVgpuVersion_t}, Ptr{nvmlVgpuVersion_t}),
                   supported, current)
end

@checked function nvmlSetVgpuVersion(vgpuVersion)
    initialize_context()
    ccall((:nvmlSetVgpuVersion, libnvml()), nvmlReturn_t,
                   (Ptr{nvmlVgpuVersion_t},),
                   vgpuVersion)
end

@checked function nvmlDeviceGetVgpuUtilization(device, lastSeenTimeStamp, sampleValType,
                                               vgpuInstanceSamplesCount, utilizationSamples)
    initialize_context()
    ccall((:nvmlDeviceGetVgpuUtilization, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Culonglong, Ptr{nvmlValueType_t}, Ptr{UInt32},
                    Ptr{nvmlVgpuInstanceUtilizationSample_t}),
                   device, lastSeenTimeStamp, sampleValType, vgpuInstanceSamplesCount,
                   utilizationSamples)
end

@checked function nvmlDeviceGetVgpuProcessUtilization(device, lastSeenTimeStamp,
                                                      vgpuProcessSamplesCount,
                                                      utilizationSamples)
    initialize_context()
    ccall((:nvmlDeviceGetVgpuProcessUtilization, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Culonglong, Ptr{UInt32},
                    Ptr{nvmlVgpuProcessUtilizationSample_t}),
                   device, lastSeenTimeStamp, vgpuProcessSamplesCount, utilizationSamples)
end

@checked function nvmlVgpuInstanceGetAccountingMode(vgpuInstance, mode)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetAccountingMode, libnvml()), nvmlReturn_t,
                   (nvmlVgpuInstance_t, Ptr{nvmlEnableState_t}),
                   vgpuInstance, mode)
end

@checked function nvmlVgpuInstanceGetAccountingPids(vgpuInstance, count, pids)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetAccountingPids, libnvml()), nvmlReturn_t,
                   (nvmlVgpuInstance_t, Ptr{UInt32}, Ptr{UInt32}),
                   vgpuInstance, count, pids)
end

@checked function nvmlVgpuInstanceGetAccountingStats(vgpuInstance, pid, stats)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetAccountingStats, libnvml()), nvmlReturn_t,
                   (nvmlVgpuInstance_t, UInt32, Ptr{nvmlAccountingStats_t}),
                   vgpuInstance, pid, stats)
end

@checked function nvmlVgpuInstanceClearAccountingPids(vgpuInstance)
    initialize_context()
    ccall((:nvmlVgpuInstanceClearAccountingPids, libnvml()), nvmlReturn_t,
                   (nvmlVgpuInstance_t,),
                   vgpuInstance)
end

@checked function nvmlGetBlacklistDeviceCount(deviceCount)
    initialize_context()
    ccall((:nvmlGetBlacklistDeviceCount, libnvml()), nvmlReturn_t,
                   (Ptr{UInt32},),
                   deviceCount)
end

@checked function nvmlGetBlacklistDeviceInfoByIndex(index, info)
    initialize_context()
    ccall((:nvmlGetBlacklistDeviceInfoByIndex, libnvml()), nvmlReturn_t,
                   (UInt32, Ptr{nvmlBlacklistDeviceInfo_t}),
                   index, info)
end

@checked function nvmlDeviceSetMigMode(device, mode, activationStatus)
    initialize_context()
    ccall((:nvmlDeviceSetMigMode, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, Ptr{nvmlReturn_t}),
                   device, mode, activationStatus)
end

@checked function nvmlDeviceGetMigMode(device, currentMode, pendingMode)
    initialize_context()
    ccall((:nvmlDeviceGetMigMode, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}, Ptr{UInt32}),
                   device, currentMode, pendingMode)
end

@checked function nvmlDeviceGetGpuInstanceProfileInfo(device, profile, info)
    initialize_context()
    ccall((:nvmlDeviceGetGpuInstanceProfileInfo, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, Ptr{nvmlGpuInstanceProfileInfo_t}),
                   device, profile, info)
end

@checked function nvmlDeviceGetGpuInstancePossiblePlacements(device, profileId, placements,
                                                             count)
    initialize_context()
    ccall((:nvmlDeviceGetGpuInstancePossiblePlacements, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, Ptr{nvmlGpuInstancePlacement_t}, Ptr{UInt32}),
                   device, profileId, placements, count)
end

@checked function nvmlDeviceGetGpuInstanceRemainingCapacity(device, profileId, count)
    initialize_context()
    ccall((:nvmlDeviceGetGpuInstanceRemainingCapacity, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, Ptr{UInt32}),
                   device, profileId, count)
end

@checked function nvmlDeviceCreateGpuInstance(device, profileId, gpuInstance)
    initialize_context()
    ccall((:nvmlDeviceCreateGpuInstance, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, Ptr{nvmlGpuInstance_t}),
                   device, profileId, gpuInstance)
end

@checked function nvmlGpuInstanceDestroy(gpuInstance)
    initialize_context()
    ccall((:nvmlGpuInstanceDestroy, libnvml()), nvmlReturn_t,
                   (nvmlGpuInstance_t,),
                   gpuInstance)
end

@checked function nvmlDeviceGetGpuInstances(device, profileId, gpuInstances, count)
    initialize_context()
    ccall((:nvmlDeviceGetGpuInstances, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, Ptr{nvmlGpuInstance_t}, Ptr{UInt32}),
                   device, profileId, gpuInstances, count)
end

@checked function nvmlDeviceGetGpuInstanceById(device, id, gpuInstance)
    initialize_context()
    ccall((:nvmlDeviceGetGpuInstanceById, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, Ptr{nvmlGpuInstance_t}),
                   device, id, gpuInstance)
end

@checked function nvmlGpuInstanceGetInfo(gpuInstance, info)
    initialize_context()
    ccall((:nvmlGpuInstanceGetInfo, libnvml()), nvmlReturn_t,
                   (nvmlGpuInstance_t, Ptr{nvmlGpuInstanceInfo_t}),
                   gpuInstance, info)
end

@checked function nvmlGpuInstanceGetComputeInstanceProfileInfo(gpuInstance, profile,
                                                               engProfile, info)
    initialize_context()
    ccall((:nvmlGpuInstanceGetComputeInstanceProfileInfo, libnvml()), nvmlReturn_t,
                   (nvmlGpuInstance_t, UInt32, UInt32,
                    Ptr{nvmlComputeInstanceProfileInfo_t}),
                   gpuInstance, profile, engProfile, info)
end

@checked function nvmlGpuInstanceGetComputeInstanceRemainingCapacity(gpuInstance,
                                                                     profileId, count)
    initialize_context()
    ccall((:nvmlGpuInstanceGetComputeInstanceRemainingCapacity, libnvml()), nvmlReturn_t,
                   (nvmlGpuInstance_t, UInt32, Ptr{UInt32}),
                   gpuInstance, profileId, count)
end

@checked function nvmlGpuInstanceCreateComputeInstance(gpuInstance, profileId,
                                                       computeInstance)
    initialize_context()
    ccall((:nvmlGpuInstanceCreateComputeInstance, libnvml()), nvmlReturn_t,
                   (nvmlGpuInstance_t, UInt32, Ptr{nvmlComputeInstance_t}),
                   gpuInstance, profileId, computeInstance)
end

@checked function nvmlComputeInstanceDestroy(computeInstance)
    initialize_context()
    ccall((:nvmlComputeInstanceDestroy, libnvml()), nvmlReturn_t,
                   (nvmlComputeInstance_t,),
                   computeInstance)
end

@checked function nvmlGpuInstanceGetComputeInstances(gpuInstance, profileId,
                                                     computeInstances, count)
    initialize_context()
    ccall((:nvmlGpuInstanceGetComputeInstances, libnvml()), nvmlReturn_t,
                   (nvmlGpuInstance_t, UInt32, Ptr{nvmlComputeInstance_t}, Ptr{UInt32}),
                   gpuInstance, profileId, computeInstances, count)
end

@checked function nvmlGpuInstanceGetComputeInstanceById(gpuInstance, id, computeInstance)
    initialize_context()
    ccall((:nvmlGpuInstanceGetComputeInstanceById, libnvml()), nvmlReturn_t,
                   (nvmlGpuInstance_t, UInt32, Ptr{nvmlComputeInstance_t}),
                   gpuInstance, id, computeInstance)
end

@checked function nvmlDeviceIsMigDeviceHandle(device, isMigDevice)
    initialize_context()
    ccall((:nvmlDeviceIsMigDeviceHandle, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}),
                   device, isMigDevice)
end

@checked function nvmlDeviceGetGpuInstanceId(device, id)
    initialize_context()
    ccall((:nvmlDeviceGetGpuInstanceId, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}),
                   device, id)
end

@checked function nvmlDeviceGetComputeInstanceId(device, id)
    initialize_context()
    ccall((:nvmlDeviceGetComputeInstanceId, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}),
                   device, id)
end

@checked function nvmlDeviceGetMaxMigDeviceCount(device, count)
    initialize_context()
    ccall((:nvmlDeviceGetMaxMigDeviceCount, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}),
                   device, count)
end

@checked function nvmlDeviceGetMigDeviceHandleByIndex(device, index, migDevice)
    initialize_context()
    ccall((:nvmlDeviceGetMigDeviceHandleByIndex, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, UInt32, Ptr{nvmlDevice_t}),
                   device, index, migDevice)
end

@checked function nvmlDeviceGetDeviceHandleFromMigDeviceHandle(migDevice, device)
    initialize_context()
    ccall((:nvmlDeviceGetDeviceHandleFromMigDeviceHandle, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlDevice_t}),
                   migDevice, device)
end

## Added in CUDA 11.1

@checked function nvmlDeviceGetGraphicsRunningProcesses_v2(device, infoCount, infos)
    initialize_context()
    ccall((:nvmlDeviceGetGraphicsRunningProcesses_v2, libnvml()), nvmlReturn_t, (nvmlDevice_t, Ptr{UInt32}, Ptr{nvmlProcessInfo_t}), device, infoCount, infos)
end

@checked function nvmlVgpuTypeGetGpuInstanceProfileId(vgpuTypeId, gpuInstanceProfileId)
    initialize_context()
    ccall((:nvmlVgpuTypeGetGpuInstanceProfileId, libnvml()), nvmlReturn_t, (nvmlVgpuTypeId_t, Ptr{UInt32}), vgpuTypeId, gpuInstanceProfileId)
end

@checked function nvmlDeviceGetAttributes_v2(device, attributes)
    initialize_context()
    ccall((:nvmlDeviceGetAttributes_v2, libnvml()), nvmlReturn_t, (nvmlDevice_t, Ptr{nvmlDeviceAttributes_t}), device, attributes)
end

@checked function nvmlDeviceGetRowRemapperHistogram(device, values)
    initialize_context()
    ccall((:nvmlDeviceGetRowRemapperHistogram, libnvml()), nvmlReturn_t, (nvmlDevice_t, Ptr{nvmlRowRemapperHistogramValues_t}), device, values)
end

@checked function nvmlDeviceGetComputeRunningProcesses_v2(device, infoCount, infos)
    initialize_context()
    ccall((:nvmlDeviceGetComputeRunningProcesses_v2, libnvml()), nvmlReturn_t, (nvmlDevice_t, Ptr{UInt32}, Ptr{nvmlProcessInfo_t}), device, infoCount, infos)
end

## Added in CUDA 11.2

@checked function nvmlComputeInstanceGetInfo_v2(computeInstance, info)
    initialize_context()
    ccall((:nvmlComputeInstanceGetInfo_v2, libnvml()), nvmlReturn_t, (nvmlComputeInstance_t, Ptr{nvmlComputeInstanceInfo_t}), computeInstance, info)
end

@checked function nvmlVgpuInstanceGetGpuInstanceId(vgpuInstance, gpuInstanceId)
    initialize_context()
    ccall((:nvmlVgpuInstanceGetGpuInstanceId, libnvml()), nvmlReturn_t, (nvmlVgpuInstance_t, Ptr{UInt32}), vgpuInstance, gpuInstanceId)
end

@checked function nvmlDeviceSetTemperatureThreshold(device, thresholdType, temp)
    initialize_context()
    ccall((:nvmlDeviceSetTemperatureThreshold, libnvml()), nvmlReturn_t, (nvmlDevice_t, nvmlTemperatureThresholds_t, Ptr{Cint}), device, thresholdType, temp)
end

## Added in CUDA 11.2 Update 2

@checked function nvmlDeviceCreateGpuInstanceWithPlacement(device, profileId, placement, gpuInstance)
    initialize_context()
    ccall((:nvmlDeviceCreateGpuInstanceWithPlacement, libnvml()), nvmlReturn_t, (nvmlDevice_t, UInt32, Ptr{nvmlGpuInstancePlacement_t}, Ptr{nvmlGpuInstance_t}), device, profileId, placement, gpuInstance)
end

##
