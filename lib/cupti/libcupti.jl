# Julia wrapper for header: cupti.h
# Automatically generated using Clang.jl

@checked function cuptiGetResultString(result, str)
    ccall((:cuptiGetResultString, libcupti()), CUptiResult,
                   (CUptiResult, Ptr{Cstring}),
                   result, str)
end

@checked function cuptiGetVersion(version)
    ccall((:cuptiGetVersion, libcupti()), CUptiResult,
                   (Ptr{UInt32},),
                   version)
end

@checked function cuptiSupportedDomains(domainCount, domainTable)
    initialize_context()
    ccall((:cuptiSupportedDomains, libcupti()), CUptiResult,
                   (Ptr{Csize_t}, Ptr{CUpti_DomainTable}),
                   domainCount, domainTable)
end

@checked function cuptiSubscribe(subscriber, callback, userdata)
    initialize_context()
    ccall((:cuptiSubscribe, libcupti()), CUptiResult,
                   (Ptr{CUpti_SubscriberHandle}, CUpti_CallbackFunc, Ptr{Cvoid}),
                   subscriber, callback, userdata)
end

@checked function cuptiUnsubscribe(subscriber)
    initialize_context()
    ccall((:cuptiUnsubscribe, libcupti()), CUptiResult,
                   (CUpti_SubscriberHandle,),
                   subscriber)
end

@checked function cuptiGetCallbackState(enable, subscriber, domain, cbid)
    initialize_context()
    ccall((:cuptiGetCallbackState, libcupti()), CUptiResult,
                   (Ptr{UInt32}, CUpti_SubscriberHandle, CUpti_CallbackDomain,
                    CUpti_CallbackId),
                   enable, subscriber, domain, cbid)
end

@checked function cuptiEnableCallback(enable, subscriber, domain, cbid)
    initialize_context()
    ccall((:cuptiEnableCallback, libcupti()), CUptiResult,
                   (UInt32, CUpti_SubscriberHandle, CUpti_CallbackDomain, CUpti_CallbackId),
                   enable, subscriber, domain, cbid)
end

@checked function cuptiEnableDomain(enable, subscriber, domain)
    initialize_context()
    ccall((:cuptiEnableDomain, libcupti()), CUptiResult,
                   (UInt32, CUpti_SubscriberHandle, CUpti_CallbackDomain),
                   enable, subscriber, domain)
end

@checked function cuptiEnableAllDomains(enable, subscriber)
    initialize_context()
    ccall((:cuptiEnableAllDomains, libcupti()), CUptiResult,
                   (UInt32, CUpti_SubscriberHandle),
                   enable, subscriber)
end

@checked function cuptiGetCallbackName(domain, cbid, name)
    initialize_context()
    ccall((:cuptiGetCallbackName, libcupti()), CUptiResult,
                   (CUpti_CallbackDomain, UInt32, Ptr{Cstring}),
                   domain, cbid, name)
end

@checked function cuptiSetEventCollectionMode(context, mode)
    initialize_context()
    ccall((:cuptiSetEventCollectionMode, libcupti()), CUptiResult,
                   (CUcontext, CUpti_EventCollectionMode),
                   context, mode)
end

@checked function cuptiDeviceGetAttribute(device, attrib, valueSize, value)
    initialize_context()
    ccall((:cuptiDeviceGetAttribute, libcupti()), CUptiResult,
                   (CUdevice, CUpti_DeviceAttribute, Ptr{Csize_t}, Ptr{Cvoid}),
                   device, attrib, valueSize, value)
end

@checked function cuptiDeviceGetTimestamp(context, timestamp)
    initialize_context()
    ccall((:cuptiDeviceGetTimestamp, libcupti()), CUptiResult,
                   (CUcontext, Ptr{UInt64}),
                   context, timestamp)
end

@checked function cuptiDeviceGetNumEventDomains(device, numDomains)
    initialize_context()
    ccall((:cuptiDeviceGetNumEventDomains, libcupti()), CUptiResult,
                   (CUdevice, Ptr{UInt32}),
                   device, numDomains)
end

@checked function cuptiDeviceEnumEventDomains(device, arraySizeBytes, domainArray)
    initialize_context()
    ccall((:cuptiDeviceEnumEventDomains, libcupti()), CUptiResult,
                   (CUdevice, Ptr{Csize_t}, Ptr{CUpti_EventDomainID}),
                   device, arraySizeBytes, domainArray)
end

@checked function cuptiDeviceGetEventDomainAttribute(device, eventDomain, attrib,
                                                     valueSize, value)
    initialize_context()
    ccall((:cuptiDeviceGetEventDomainAttribute, libcupti()), CUptiResult,
                   (CUdevice, CUpti_EventDomainID, CUpti_EventDomainAttribute,
                    Ptr{Csize_t}, Ptr{Cvoid}),
                   device, eventDomain, attrib, valueSize, value)
end

@checked function cuptiGetNumEventDomains(numDomains)
    initialize_context()
    ccall((:cuptiGetNumEventDomains, libcupti()), CUptiResult,
                   (Ptr{UInt32},),
                   numDomains)
end

@checked function cuptiEnumEventDomains(arraySizeBytes, domainArray)
    initialize_context()
    ccall((:cuptiEnumEventDomains, libcupti()), CUptiResult,
                   (Ptr{Csize_t}, Ptr{CUpti_EventDomainID}),
                   arraySizeBytes, domainArray)
end

@checked function cuptiEventDomainGetAttribute(eventDomain, attrib, valueSize, value)
    initialize_context()
    ccall((:cuptiEventDomainGetAttribute, libcupti()), CUptiResult,
                   (CUpti_EventDomainID, CUpti_EventDomainAttribute, Ptr{Csize_t},
                    Ptr{Cvoid}),
                   eventDomain, attrib, valueSize, value)
end

@checked function cuptiEventDomainGetNumEvents(eventDomain, numEvents)
    initialize_context()
    ccall((:cuptiEventDomainGetNumEvents, libcupti()), CUptiResult,
                   (CUpti_EventDomainID, Ptr{UInt32}),
                   eventDomain, numEvents)
end

@checked function cuptiEventDomainEnumEvents(eventDomain, arraySizeBytes, eventArray)
    initialize_context()
    ccall((:cuptiEventDomainEnumEvents, libcupti()), CUptiResult,
                   (CUpti_EventDomainID, Ptr{Csize_t}, Ptr{CUpti_EventID}),
                   eventDomain, arraySizeBytes, eventArray)
end

@checked function cuptiEventGetAttribute(event, attrib, valueSize, value)
    initialize_context()
    ccall((:cuptiEventGetAttribute, libcupti()), CUptiResult,
                   (CUpti_EventID, CUpti_EventAttribute, Ptr{Csize_t}, Ptr{Cvoid}),
                   event, attrib, valueSize, value)
end

@checked function cuptiEventGetIdFromName(device, eventName, event)
    initialize_context()
    ccall((:cuptiEventGetIdFromName, libcupti()), CUptiResult,
                   (CUdevice, Cstring, Ptr{CUpti_EventID}),
                   device, eventName, event)
end

@checked function cuptiEventGroupCreate(context, eventGroup, flags)
    initialize_context()
    ccall((:cuptiEventGroupCreate, libcupti()), CUptiResult,
                   (CUcontext, Ptr{CUpti_EventGroup}, UInt32),
                   context, eventGroup, flags)
end

@checked function cuptiEventGroupDestroy(eventGroup)
    initialize_context()
    ccall((:cuptiEventGroupDestroy, libcupti()), CUptiResult,
                   (CUpti_EventGroup,),
                   eventGroup)
end

@checked function cuptiEventGroupGetAttribute(eventGroup, attrib, valueSize, value)
    initialize_context()
    ccall((:cuptiEventGroupGetAttribute, libcupti()), CUptiResult,
                   (CUpti_EventGroup, CUpti_EventGroupAttribute, Ptr{Csize_t}, Ptr{Cvoid}),
                   eventGroup, attrib, valueSize, value)
end

@checked function cuptiEventGroupSetAttribute(eventGroup, attrib, valueSize, value)
    initialize_context()
    ccall((:cuptiEventGroupSetAttribute, libcupti()), CUptiResult,
                   (CUpti_EventGroup, CUpti_EventGroupAttribute, Csize_t, Ptr{Cvoid}),
                   eventGroup, attrib, valueSize, value)
end

@checked function cuptiEventGroupAddEvent(eventGroup, event)
    initialize_context()
    ccall((:cuptiEventGroupAddEvent, libcupti()), CUptiResult,
                   (CUpti_EventGroup, CUpti_EventID),
                   eventGroup, event)
end

@checked function cuptiEventGroupRemoveEvent(eventGroup, event)
    initialize_context()
    ccall((:cuptiEventGroupRemoveEvent, libcupti()), CUptiResult,
                   (CUpti_EventGroup, CUpti_EventID),
                   eventGroup, event)
end

@checked function cuptiEventGroupRemoveAllEvents(eventGroup)
    initialize_context()
    ccall((:cuptiEventGroupRemoveAllEvents, libcupti()), CUptiResult,
                   (CUpti_EventGroup,),
                   eventGroup)
end

@checked function cuptiEventGroupResetAllEvents(eventGroup)
    initialize_context()
    ccall((:cuptiEventGroupResetAllEvents, libcupti()), CUptiResult,
                   (CUpti_EventGroup,),
                   eventGroup)
end

@checked function cuptiEventGroupEnable(eventGroup)
    initialize_context()
    ccall((:cuptiEventGroupEnable, libcupti()), CUptiResult,
                   (CUpti_EventGroup,),
                   eventGroup)
end

@checked function cuptiEventGroupDisable(eventGroup)
    initialize_context()
    ccall((:cuptiEventGroupDisable, libcupti()), CUptiResult,
                   (CUpti_EventGroup,),
                   eventGroup)
end

@checked function cuptiEventGroupReadEvent(eventGroup, flags, event,
                                           eventValueBufferSizeBytes, eventValueBuffer)
    initialize_context()
    ccall((:cuptiEventGroupReadEvent, libcupti()), CUptiResult,
                   (CUpti_EventGroup, CUpti_ReadEventFlags, CUpti_EventID, Ptr{Csize_t},
                    Ptr{UInt64}),
                   eventGroup, flags, event, eventValueBufferSizeBytes, eventValueBuffer)
end

@checked function cuptiEventGroupReadAllEvents(eventGroup, flags,
                                               eventValueBufferSizeBytes, eventValueBuffer,
                                               eventIdArraySizeBytes, eventIdArray,
                                               numEventIdsRead)
    initialize_context()
    ccall((:cuptiEventGroupReadAllEvents, libcupti()), CUptiResult,
                   (CUpti_EventGroup, CUpti_ReadEventFlags, Ptr{Csize_t}, Ptr{UInt64},
                    Ptr{Csize_t}, Ptr{CUpti_EventID}, Ptr{Csize_t}),
                   eventGroup, flags, eventValueBufferSizeBytes, eventValueBuffer,
                   eventIdArraySizeBytes, eventIdArray, numEventIdsRead)
end

@checked function cuptiEventGroupSetsCreate(context, eventIdArraySizeBytes, eventIdArray,
                                            eventGroupPasses)
    initialize_context()
    ccall((:cuptiEventGroupSetsCreate, libcupti()), CUptiResult,
                   (CUcontext, Csize_t, Ptr{CUpti_EventID}, Ptr{Ptr{CUpti_EventGroupSets}}),
                   context, eventIdArraySizeBytes, eventIdArray, eventGroupPasses)
end

@checked function cuptiEventGroupSetsDestroy(eventGroupSets)
    initialize_context()
    ccall((:cuptiEventGroupSetsDestroy, libcupti()), CUptiResult,
                   (Ptr{CUpti_EventGroupSets},),
                   eventGroupSets)
end

@checked function cuptiEventGroupSetEnable(eventGroupSet)
    initialize_context()
    ccall((:cuptiEventGroupSetEnable, libcupti()), CUptiResult,
                   (Ptr{CUpti_EventGroupSet},),
                   eventGroupSet)
end

@checked function cuptiEventGroupSetDisable(eventGroupSet)
    initialize_context()
    ccall((:cuptiEventGroupSetDisable, libcupti()), CUptiResult,
                   (Ptr{CUpti_EventGroupSet},),
                   eventGroupSet)
end

@checked function cuptiEnableKernelReplayMode(context)
    initialize_context()
    ccall((:cuptiEnableKernelReplayMode, libcupti()), CUptiResult,
                   (CUcontext,),
                   context)
end

@checked function cuptiDisableKernelReplayMode(context)
    initialize_context()
    ccall((:cuptiDisableKernelReplayMode, libcupti()), CUptiResult,
                   (CUcontext,),
                   context)
end

@checked function cuptiKernelReplaySubscribeUpdate(updateFunc, customData)
    initialize_context()
    ccall((:cuptiKernelReplaySubscribeUpdate, libcupti()), CUptiResult,
                   (CUpti_KernelReplayUpdateFunc, Ptr{Cvoid}),
                   updateFunc, customData)
end

@checked function cuptiGetNumMetrics(numMetrics)
    initialize_context()
    ccall((:cuptiGetNumMetrics, libcupti()), CUptiResult,
                   (Ptr{UInt32},),
                   numMetrics)
end

@checked function cuptiEnumMetrics(arraySizeBytes, metricArray)
    initialize_context()
    ccall((:cuptiEnumMetrics, libcupti()), CUptiResult,
                   (Ptr{Csize_t}, Ptr{CUpti_MetricID}),
                   arraySizeBytes, metricArray)
end

@checked function cuptiDeviceGetNumMetrics(device, numMetrics)
    initialize_context()
    ccall((:cuptiDeviceGetNumMetrics, libcupti()), CUptiResult,
                   (CUdevice, Ptr{UInt32}),
                   device, numMetrics)
end

@checked function cuptiDeviceEnumMetrics(device, arraySizeBytes, metricArray)
    initialize_context()
    ccall((:cuptiDeviceEnumMetrics, libcupti()), CUptiResult,
                   (CUdevice, Ptr{Csize_t}, Ptr{CUpti_MetricID}),
                   device, arraySizeBytes, metricArray)
end

@checked function cuptiMetricGetAttribute(metric, attrib, valueSize, value)
    initialize_context()
    ccall((:cuptiMetricGetAttribute, libcupti()), CUptiResult,
                   (CUpti_MetricID, CUpti_MetricAttribute, Ptr{Csize_t}, Ptr{Cvoid}),
                   metric, attrib, valueSize, value)
end

@checked function cuptiMetricGetIdFromName(device, metricName, metric)
    initialize_context()
    ccall((:cuptiMetricGetIdFromName, libcupti()), CUptiResult,
                   (CUdevice, Cstring, Ptr{CUpti_MetricID}),
                   device, metricName, metric)
end

@checked function cuptiMetricGetNumEvents(metric, numEvents)
    initialize_context()
    ccall((:cuptiMetricGetNumEvents, libcupti()), CUptiResult,
                   (CUpti_MetricID, Ptr{UInt32}),
                   metric, numEvents)
end

@checked function cuptiMetricEnumEvents(metric, eventIdArraySizeBytes, eventIdArray)
    initialize_context()
    ccall((:cuptiMetricEnumEvents, libcupti()), CUptiResult,
                   (CUpti_MetricID, Ptr{Csize_t}, Ptr{CUpti_EventID}),
                   metric, eventIdArraySizeBytes, eventIdArray)
end

@checked function cuptiMetricGetNumProperties(metric, numProp)
    initialize_context()
    ccall((:cuptiMetricGetNumProperties, libcupti()), CUptiResult,
                   (CUpti_MetricID, Ptr{UInt32}),
                   metric, numProp)
end

@checked function cuptiMetricEnumProperties(metric, propIdArraySizeBytes, propIdArray)
    initialize_context()
    ccall((:cuptiMetricEnumProperties, libcupti()), CUptiResult,
                   (CUpti_MetricID, Ptr{Csize_t}, Ptr{CUpti_MetricPropertyID}),
                   metric, propIdArraySizeBytes, propIdArray)
end

@checked function cuptiMetricGetRequiredEventGroupSets(context, metric, eventGroupSets)
    initialize_context()
    ccall((:cuptiMetricGetRequiredEventGroupSets, libcupti()), CUptiResult,
                   (CUcontext, CUpti_MetricID, Ptr{Ptr{CUpti_EventGroupSets}}),
                   context, metric, eventGroupSets)
end

@checked function cuptiMetricCreateEventGroupSets(context, metricIdArraySizeBytes,
                                                  metricIdArray, eventGroupPasses)
    initialize_context()
    ccall((:cuptiMetricCreateEventGroupSets, libcupti()), CUptiResult,
                   (CUcontext, Csize_t, Ptr{CUpti_MetricID},
                    Ptr{Ptr{CUpti_EventGroupSets}}),
                   context, metricIdArraySizeBytes, metricIdArray, eventGroupPasses)
end

@checked function cuptiMetricGetValue(device, metric, eventIdArraySizeBytes, eventIdArray,
                                      eventValueArraySizeBytes, eventValueArray,
                                      timeDuration, metricValue)
    initialize_context()
    ccall((:cuptiMetricGetValue, libcupti()), CUptiResult,
                   (CUdevice, CUpti_MetricID, Csize_t, Ptr{CUpti_EventID}, Csize_t,
                    Ptr{UInt64}, UInt64, Ptr{CUpti_MetricValue}),
                   device, metric, eventIdArraySizeBytes, eventIdArray,
                   eventValueArraySizeBytes, eventValueArray, timeDuration, metricValue)
end

@checked function cuptiMetricGetValue2(metric, eventIdArraySizeBytes, eventIdArray,
                                       eventValueArraySizeBytes, eventValueArray,
                                       propIdArraySizeBytes, propIdArray,
                                       propValueArraySizeBytes, propValueArray, metricValue)
    initialize_context()
    ccall((:cuptiMetricGetValue2, libcupti()), CUptiResult,
                   (CUpti_MetricID, Csize_t, Ptr{CUpti_EventID}, Csize_t, Ptr{UInt64},
                    Csize_t, Ptr{CUpti_MetricPropertyID}, Csize_t, Ptr{UInt64},
                    Ptr{CUpti_MetricValue}),
                   metric, eventIdArraySizeBytes, eventIdArray, eventValueArraySizeBytes,
                   eventValueArray, propIdArraySizeBytes, propIdArray,
                   propValueArraySizeBytes, propValueArray, metricValue)
end

@checked function cuptiGetTimestamp(timestamp)
    initialize_context()
    ccall((:cuptiGetTimestamp, libcupti()), CUptiResult,
                   (Ptr{UInt64},),
                   timestamp)
end

@checked function cuptiGetContextId(context, contextId)
    initialize_context()
    ccall((:cuptiGetContextId, libcupti()), CUptiResult,
                   (CUcontext, Ptr{UInt32}),
                   context, contextId)
end

@checked function cuptiGetStreamId(context, stream, streamId)
    initialize_context()
    ccall((:cuptiGetStreamId, libcupti()), CUptiResult,
                   (CUcontext, CUstream, Ptr{UInt32}),
                   context, stream, streamId)
end

@checked function cuptiGetStreamIdEx(context, stream, perThreadStream, streamId)
    initialize_context()
    ccall((:cuptiGetStreamIdEx, libcupti()), CUptiResult,
                   (CUcontext, CUstream, UInt8, Ptr{UInt32}),
                   context, stream, perThreadStream, streamId)
end

@checked function cuptiGetDeviceId(context, deviceId)
    initialize_context()
    ccall((:cuptiGetDeviceId, libcupti()), CUptiResult,
                   (CUcontext, Ptr{UInt32}),
                   context, deviceId)
end

@checked function cuptiGetGraphNodeId(node, nodeId)
    initialize_context()
    ccall((:cuptiGetGraphNodeId, libcupti()), CUptiResult,
                   (CUgraphNode, Ptr{UInt64}),
                   node, nodeId)
end

@checked function cuptiActivityEnable(kind)
    initialize_context()
    ccall((:cuptiActivityEnable, libcupti()), CUptiResult,
                   (CUpti_ActivityKind,),
                   kind)
end

@checked function cuptiActivityDisable(kind)
    initialize_context()
    ccall((:cuptiActivityDisable, libcupti()), CUptiResult,
                   (CUpti_ActivityKind,),
                   kind)
end

@checked function cuptiActivityEnableContext(context, kind)
    initialize_context()
    ccall((:cuptiActivityEnableContext, libcupti()), CUptiResult,
                   (CUcontext, CUpti_ActivityKind),
                   context, kind)
end

@checked function cuptiActivityDisableContext(context, kind)
    initialize_context()
    ccall((:cuptiActivityDisableContext, libcupti()), CUptiResult,
                   (CUcontext, CUpti_ActivityKind),
                   context, kind)
end

@checked function cuptiActivityGetNumDroppedRecords(context, streamId, dropped)
    initialize_context()
    ccall((:cuptiActivityGetNumDroppedRecords, libcupti()), CUptiResult,
                   (CUcontext, UInt32, Ptr{Csize_t}),
                   context, streamId, dropped)
end

@checked function cuptiActivityGetNextRecord(buffer, validBufferSizeBytes, record)
    initialize_context()
    ccall((:cuptiActivityGetNextRecord, libcupti()), CUptiResult,
                   (Ptr{UInt8}, Csize_t, Ptr{Ptr{CUpti_Activity}}),
                   buffer, validBufferSizeBytes, record)
end

@checked function cuptiActivityRegisterCallbacks(funcBufferRequested, funcBufferCompleted)
    initialize_context()
    ccall((:cuptiActivityRegisterCallbacks, libcupti()), CUptiResult,
                   (CUpti_BuffersCallbackRequestFunc, CUpti_BuffersCallbackCompleteFunc),
                   funcBufferRequested, funcBufferCompleted)
end

@checked function cuptiActivityFlush(context, streamId, flag)
    initialize_context()
    ccall((:cuptiActivityFlush, libcupti()), CUptiResult,
                   (CUcontext, UInt32, UInt32),
                   context, streamId, flag)
end

@checked function cuptiActivityFlushAll(flag)
    initialize_context()
    ccall((:cuptiActivityFlushAll, libcupti()), CUptiResult,
                   (UInt32,),
                   flag)
end

@checked function cuptiActivityGetAttribute(attr, valueSize, value)
    initialize_context()
    ccall((:cuptiActivityGetAttribute, libcupti()), CUptiResult,
                   (CUpti_ActivityAttribute, Ptr{Csize_t}, Ptr{Cvoid}),
                   attr, valueSize, value)
end

@checked function cuptiActivitySetAttribute(attr, valueSize, value)
    initialize_context()
    ccall((:cuptiActivitySetAttribute, libcupti()), CUptiResult,
                   (CUpti_ActivityAttribute, Ptr{Csize_t}, Ptr{Cvoid}),
                   attr, valueSize, value)
end

@checked function cuptiActivityConfigureUnifiedMemoryCounter(config, count)
    initialize_context()
    ccall((:cuptiActivityConfigureUnifiedMemoryCounter, libcupti()), CUptiResult,
                   (Ptr{CUpti_ActivityUnifiedMemoryCounterConfig}, UInt32),
                   config, count)
end

@checked function cuptiGetAutoBoostState(context, state)
    initialize_context()
    ccall((:cuptiGetAutoBoostState, libcupti()), CUptiResult,
                   (CUcontext, Ptr{CUpti_ActivityAutoBoostState}),
                   context, state)
end

@checked function cuptiActivityConfigurePCSampling(ctx, config)
    initialize_context()
    ccall((:cuptiActivityConfigurePCSampling, libcupti()), CUptiResult,
                   (CUcontext, Ptr{CUpti_ActivityPCSamplingConfig}),
                   ctx, config)
end

@checked function cuptiGetLastError()
    initialize_context()
    ccall((:cuptiGetLastError, libcupti()), CUptiResult, ())
end

@checked function cuptiSetThreadIdType(type)
    initialize_context()
    ccall((:cuptiSetThreadIdType, libcupti()), CUptiResult,
                   (CUpti_ActivityThreadIdType,),
                   type)
end

@checked function cuptiGetThreadIdType(type)
    initialize_context()
    ccall((:cuptiGetThreadIdType, libcupti()), CUptiResult,
                   (Ptr{CUpti_ActivityThreadIdType},),
                   type)
end

@checked function cuptiComputeCapabilitySupported(major, minor, support)
    initialize_context()
    ccall((:cuptiComputeCapabilitySupported, libcupti()), CUptiResult,
                   (Cint, Cint, Ptr{Cint}),
                   major, minor, support)
end

@checked function cuptiDeviceSupported(dev, support)
    initialize_context()
    ccall((:cuptiDeviceSupported, libcupti()), CUptiResult,
                   (CUdevice, Ptr{Cint}),
                   dev, support)
end

@checked function cuptiDeviceVirtualizationMode(dev, mode)
    initialize_context()
    ccall((:cuptiDeviceVirtualizationMode, libcupti()), CUptiResult,
                   (CUdevice, Ptr{CUpti_DeviceVirtualizationMode}),
                   dev, mode)
end

@checked function cuptiFinalize()
    initialize_context()
    ccall((:cuptiFinalize, libcupti()), CUptiResult, ())
end

@checked function cuptiActivityPushExternalCorrelationId(kind, id)
    initialize_context()
    ccall((:cuptiActivityPushExternalCorrelationId, libcupti()), CUptiResult,
                   (CUpti_ExternalCorrelationKind, UInt64),
                   kind, id)
end

@checked function cuptiActivityPopExternalCorrelationId(kind, lastId)
    initialize_context()
    ccall((:cuptiActivityPopExternalCorrelationId, libcupti()), CUptiResult,
                   (CUpti_ExternalCorrelationKind, Ptr{UInt64}),
                   kind, lastId)
end

@checked function cuptiActivityEnableLatencyTimestamps(enable)
    initialize_context()
    ccall((:cuptiActivityEnableLatencyTimestamps, libcupti()), CUptiResult,
                   (UInt8,),
                   enable)
end
# Julia wrapper for header: cupti_profiler_target.h
# Automatically generated using Clang.jl

@checked function cuptiProfilerInitialize(pParams)
    initialize_context()
    ccall((:cuptiProfilerInitialize, libcupti()), CUptiResult,
                   (Ptr{CUpti_Profiler_Initialize_Params},),
                   pParams)
end

@checked function cuptiProfilerDeInitialize(pParams)
    initialize_context()
    ccall((:cuptiProfilerDeInitialize, libcupti()), CUptiResult,
                   (Ptr{CUpti_Profiler_DeInitialize_Params},),
                   pParams)
end

@checked function cuptiProfilerCounterDataImageCalculateSize(pParams)
    initialize_context()
    ccall((:cuptiProfilerCounterDataImageCalculateSize, libcupti()), CUptiResult,
                   (Ptr{CUpti_Profiler_CounterDataImage_CalculateSize_Params},),
                   pParams)
end

@checked function cuptiProfilerCounterDataImageInitialize(pParams)
    initialize_context()
    ccall((:cuptiProfilerCounterDataImageInitialize, libcupti()), CUptiResult,
                   (Ptr{CUpti_Profiler_CounterDataImage_Initialize_Params},),
                   pParams)
end

@checked function cuptiProfilerCounterDataImageCalculateScratchBufferSize(pParams)
    initialize_context()
    ccall((:cuptiProfilerCounterDataImageCalculateScratchBufferSize, libcupti()), CUptiResult,
                   (Ptr{CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params},
                    ),
                   pParams)
end

@checked function cuptiProfilerCounterDataImageInitializeScratchBuffer(pParams)
    initialize_context()
    ccall((:cuptiProfilerCounterDataImageInitializeScratchBuffer, libcupti()), CUptiResult,
                   (Ptr{CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params},),
                   pParams)
end

@checked function cuptiProfilerBeginSession(pParams)
    initialize_context()
    ccall((:cuptiProfilerBeginSession, libcupti()), CUptiResult,
                   (Ptr{CUpti_Profiler_BeginSession_Params},),
                   pParams)
end

@checked function cuptiProfilerEndSession(pParams)
    initialize_context()
    ccall((:cuptiProfilerEndSession, libcupti()), CUptiResult,
                   (Ptr{CUpti_Profiler_EndSession_Params},),
                   pParams)
end

@checked function cuptiProfilerSetConfig(pParams)
    initialize_context()
    ccall((:cuptiProfilerSetConfig, libcupti()), CUptiResult,
                   (Ptr{CUpti_Profiler_SetConfig_Params},),
                   pParams)
end

@checked function cuptiProfilerUnsetConfig(pParams)
    initialize_context()
    ccall((:cuptiProfilerUnsetConfig, libcupti()), CUptiResult,
                   (Ptr{CUpti_Profiler_UnsetConfig_Params},),
                   pParams)
end

@checked function cuptiProfilerBeginPass(pParams)
    initialize_context()
    ccall((:cuptiProfilerBeginPass, libcupti()), CUptiResult,
                   (Ptr{CUpti_Profiler_BeginPass_Params},),
                   pParams)
end

@checked function cuptiProfilerEndPass(pParams)
    initialize_context()
    ccall((:cuptiProfilerEndPass, libcupti()), CUptiResult,
                   (Ptr{CUpti_Profiler_EndPass_Params},),
                   pParams)
end

@checked function cuptiProfilerEnableProfiling(pParams)
    initialize_context()
    ccall((:cuptiProfilerEnableProfiling, libcupti()), CUptiResult,
                   (Ptr{CUpti_Profiler_EnableProfiling_Params},),
                   pParams)
end

@checked function cuptiProfilerDisableProfiling(pParams)
    initialize_context()
    ccall((:cuptiProfilerDisableProfiling, libcupti()), CUptiResult,
                   (Ptr{CUpti_Profiler_DisableProfiling_Params},),
                   pParams)
end

@checked function cuptiProfilerIsPassCollected(pParams)
    initialize_context()
    ccall((:cuptiProfilerIsPassCollected, libcupti()), CUptiResult,
                   (Ptr{CUpti_Profiler_IsPassCollected_Params},),
                   pParams)
end

@checked function cuptiProfilerFlushCounterData(pParams)
    initialize_context()
    ccall((:cuptiProfilerFlushCounterData, libcupti()), CUptiResult,
                   (Ptr{CUpti_Profiler_FlushCounterData_Params},),
                   pParams)
end

@checked function cuptiProfilerPushRange(pParams)
    initialize_context()
    ccall((:cuptiProfilerPushRange, libcupti()), CUptiResult,
                   (Ptr{CUpti_Profiler_PushRange_Params},),
                   pParams)
end

@checked function cuptiProfilerPopRange(pParams)
    initialize_context()
    ccall((:cuptiProfilerPopRange, libcupti()), CUptiResult,
                   (Ptr{CUpti_Profiler_PopRange_Params},),
                   pParams)
end

@checked function cuptiProfilerGetCounterAvailability(pParams)
    initialize_context()
    ccall((:cuptiProfilerGetCounterAvailability, libcupti()), CUptiResult,
                   (Ptr{CUpti_Profiler_GetCounterAvailability_Params},),
                   pParams)
end

## Added in CUDA 11.1

@checked function cuptiGetGraphId(graph, pId)
    initialize_context()
    ccall((:cuptiGetGraphId, libcupti()), CUptiResult, (CUgraph, Ptr{UInt32}), graph, pId)
end

@checked function cuptiActivityFlushPeriod(time)
    initialize_context()
    ccall((:cuptiActivityFlushPeriod, libcupti()), CUptiResult, (UInt32,), time)
end

## Added in CUDA 11.2

@checked function cuptiActivityEnableLaunchAttributes(enable)
    initialize_context()
    ccall((:cuptiActivityEnableLaunchAttributes, libcupti()), CUptiResult, (UInt8,), enable)
end

##
