# library and API version management

function version()
    version_ref = Ref{Cint}()
    @apicall(:cuDriverGetVersion, (Ptr{Cint},), version_ref)
    return version_ref[]
end

function api_versioning(mapping, version)
    if version >= 3020
        mapping[:cuDeviceTotalMem]           = :cuDeviceTotalMem_v2
        mapping[:cuCtxCreate]                = :cuCtxCreate_v2
        mapping[:cuModuleGetGlobal]          = :cuModuleGetGlobal_v2
        mapping[:cuMemGetInfo]               = :cuMemGetInfo_v2
        mapping[:cuMemAlloc]                 = :cuMemAlloc_v2
        mapping[:cuMemAllocPitch]            = :cuMemAllocPitch_v2
        mapping[:cuMemFree]                  = :cuMemFree_v2
        mapping[:cuMemGetAddressRange]       = :cuMemGetAddressRange_v2
        mapping[:cuMemAllocHost]             = :cuMemAllocHost_v2
        mapping[:cuMemHostGetDevicePointer]  = :cuMemHostGetDevicePointer_v2
        mapping[:cuMemcpyHtoD]               = :cuMemcpyHtoD_v2
        mapping[:cuMemcpyDtoH]               = :cuMemcpyDtoH_v2
        mapping[:cuMemcpyDtoD]               = :cuMemcpyDtoD_v2
        mapping[:cuMemcpyDtoA]               = :cuMemcpyDtoA_v2
        mapping[:cuMemcpyAtoD]               = :cuMemcpyAtoD_v2
        mapping[:cuMemcpyHtoA]               = :cuMemcpyHtoA_v2
        mapping[:cuMemcpyAtoH]               = :cuMemcpyAtoH_v2
        mapping[:cuMemcpyAtoA]               = :cuMemcpyAtoA_v2
        mapping[:cuMemcpyHtoAAsync]          = :cuMemcpyHtoAAsync_v2
        mapping[:cuMemcpyAtoHAsync]          = :cuMemcpyAtoHAsync_v2
        mapping[:cuMemcpy2D]                 = :cuMemcpy2D_v2
        mapping[:cuMemcpy2DUnaligned]        = :cuMemcpy2DUnaligned_v2
        mapping[:cuMemcpy3D]                 = :cuMemcpy3D_v2
        mapping[:cuMemcpyHtoDAsync]          = :cuMemcpyHtoDAsync_v2
        mapping[:cuMemcpyDtoHAsync]          = :cuMemcpyDtoHAsync_v2
        mapping[:cuMemcpyDtoDAsync]          = :cuMemcpyDtoDAsync_v2
        mapping[:cuMemcpy2DAsync]            = :cuMemcpy2DAsync_v2
        mapping[:cuMemcpy3DAsync]            = :cuMemcpy3DAsync_v2
        mapping[:cuMemsetD8]                 = :cuMemsetD8_v2
        mapping[:cuMemsetD16]                = :cuMemsetD16_v2
        mapping[:cuMemsetD32]                = :cuMemsetD32_v2
        mapping[:cuMemsetD2D8]               = :cuMemsetD2D8_v2
        mapping[:cuMemsetD2D16]              = :cuMemsetD2D16_v2
        mapping[:cuMemsetD2D32]              = :cuMemsetD2D32_v2
        mapping[:cuArrayCreate]              = :cuArrayCreate_v2
        mapping[:cuArrayGetDescriptor]       = :cuArrayGetDescriptor_v2
        mapping[:cuArray3DCreate]            = :cuArray3DCreate_v2
        mapping[:cuArray3DGetDescriptor]     = :cuArray3DGetDescriptor_v2
        mapping[:cuTexRefSetAddress]         = :cuTexRefSetAddress_v2
        mapping[:cuTexRefGetAddress]         = :cuTexRefGetAddress_v2
        mapping[:cuGraphicsResourceGetMappedPointer] = :cuGraphicsResourceGetMappedPointer_v2
        mapping[:cuDeviceTotalMem]           = :cuDeviceTotalMem_v2
        mapping[:cuCtxCreate]                = :cuCtxCreate_v2
        mapping[:cuMemAlloc]                 = :cuMemAlloc_v2
        mapping[:cuMemcpyHtoD]               = :cuMemcpyHtoD_v2
        mapping[:cuMemcpyDtoH]               = :cuMemcpyDtoH_v2
        mapping[:cuMemFree]                  = :cuMemFree_v2
        mapping[:cuModuleGetGlobal]          = :cuModuleGetGlobal_v2
        mapping[:cuMemsetD32]                = :cuMemsetD32_v2
    end
    if version >= 4000
        mapping[:cuCtxDestroy]               = :cuCtxDestroy_v2
        mapping[:cuCtxPushCurrent]           = :cuCtxPushCurrent_v2
        mapping[:cuCtxPopCurrent]            = :cuCtxPopCurrent_v2
    end
end
