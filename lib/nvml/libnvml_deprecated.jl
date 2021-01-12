## Deprecated in CUDA 11.1

struct nvmlDeviceAttributesV1_st
    multiprocessorCount::UInt32
    sharedCopyEngineCount::UInt32
    sharedDecoderCount::UInt32
    sharedEncoderCount::UInt32
    sharedJpegCount::UInt32
    sharedOfaCount::UInt32
end

const nvmlDeviceAttributesV1_t = nvmlDeviceAttributesV1_st

@checked function nvmlDeviceGetAttributes(device, attributes)
    initialize_api()
    ccall((:nvmlDeviceGetAttributes, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlDeviceAttributesV1_t}),
                   device, attributes)
end

@checked function nvmlDeviceGetAttributes(device, attributes)
    initialize_api()
    ccall((:nvmlDeviceGetAttributes, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{nvmlDeviceAttributesV1_t}),
                   device, attributes)
end

struct nvmlProcessInfoV1_st
    pid::UInt32
    usedGpuMemory::Culonglong
end

const nvmlProcessInfoV1_t = nvmlProcessInfoV1_st

@checked function nvmlDeviceGetComputeRunningProcesses(device, infoCount, infos)
    initialize_api()
    ccall((:nvmlDeviceGetComputeRunningProcesses, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}, Ptr{nvmlProcessInfoV1_t}),
                   device, infoCount, infos)
end

@checked function nvmlDeviceGetGraphicsRunningProcesses(device, infoCount, infos)
    initialize_api()
    ccall((:nvmlDeviceGetGraphicsRunningProcesses, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}, Ptr{nvmlProcessInfoV1_t}),
                   device, infoCount, infos)
end

@checked function nvmlDeviceGetComputeRunningProcesses(device, infoCount, infos)
    initialize_api()
    ccall((:nvmlDeviceGetComputeRunningProcesses, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}, Ptr{nvmlProcessInfoV1_t}),
                   device, infoCount, infos)
end

@checked function nvmlDeviceGetGraphicsRunningProcesses(device, infoCount, infos)
    initialize_api()
    ccall((:nvmlDeviceGetGraphicsRunningProcesses, libnvml()), nvmlReturn_t,
                   (nvmlDevice_t, Ptr{UInt32}, Ptr{nvmlProcessInfoV1_t}),
                   device, infoCount, infos)
end

##
