# Julia wrapper for header: cuda.h
# Automatically generated using Clang.jl

@checked function cuGetErrorString(error, pStr)
    ccall((:cuGetErrorString, libcuda()), CUresult,
                   (CUresult, Ptr{Cstring}),
                   error, pStr)
end

@checked function cuGetErrorName(error, pStr)
    ccall((:cuGetErrorName, libcuda()), CUresult,
                   (CUresult, Ptr{Cstring}),
                   error, pStr)
end

@checked function cuInit(Flags)
    ccall((:cuInit, libcuda()), CUresult,
                   (UInt32,),
                   Flags)
end

@checked function cuDriverGetVersion(driverVersion)
    ccall((:cuDriverGetVersion, libcuda()), CUresult,
                   (Ptr{Cint},),
                   driverVersion)
end

@checked function cuDeviceGet(device, ordinal)
    ccall((:cuDeviceGet, libcuda()), CUresult,
                   (Ptr{CUdevice}, Cint),
                   device, ordinal)
end

@checked function cuDeviceGetCount(count)
    ccall((:cuDeviceGetCount, libcuda()), CUresult,
                   (Ptr{Cint},),
                   count)
end

@checked function cuDeviceGetName(name, len, dev)
    ccall((:cuDeviceGetName, libcuda()), CUresult,
                   (Cstring, Cint, CUdevice),
                   name, len, dev)
end

@checked function cuDeviceGetUuid(uuid, dev)
    ccall((:cuDeviceGetUuid, libcuda()), CUresult,
                   (Ptr{CUuuid}, CUdevice),
                   uuid, dev)
end

@checked function cuDeviceTotalMem_v2(bytes, dev)
    ccall((:cuDeviceTotalMem_v2, libcuda()), CUresult,
                   (Ptr{Csize_t}, CUdevice),
                   bytes, dev)
end

@checked function cuDeviceGetAttribute(pi, attrib, dev)
    ccall((:cuDeviceGetAttribute, libcuda()), CUresult,
                   (Ptr{Cint}, CUdevice_attribute, CUdevice),
                   pi, attrib, dev)
end

@checked function cuDeviceGetNvSciSyncAttributes(nvSciSyncAttrList, dev, flags)
    initialize_api()
    ccall((:cuDeviceGetNvSciSyncAttributes, libcuda()), CUresult,
                   (Ptr{Cvoid}, CUdevice, Cint),
                   nvSciSyncAttrList, dev, flags)
end

@checked function cuDeviceGetProperties(prop, dev)
    ccall((:cuDeviceGetProperties, libcuda()), CUresult,
                   (Ptr{CUdevprop}, CUdevice),
                   prop, dev)
end

@checked function cuDeviceComputeCapability(major, minor, dev)
    ccall((:cuDeviceComputeCapability, libcuda()), CUresult,
                   (Ptr{Cint}, Ptr{Cint}, CUdevice),
                   major, minor, dev)
end

@checked function cuDevicePrimaryCtxRetain(pctx, dev)
    ccall((:cuDevicePrimaryCtxRetain, libcuda()), CUresult,
                   (Ptr{CUcontext}, CUdevice),
                   pctx, dev)
end

@checked function cuDevicePrimaryCtxRelease_v2(dev)
    ccall((:cuDevicePrimaryCtxRelease_v2, libcuda()), CUresult,
                   (CUdevice,),
                   dev)
end

@checked function cuDevicePrimaryCtxSetFlags_v2(dev, flags)
    ccall((:cuDevicePrimaryCtxSetFlags_v2, libcuda()), CUresult,
                   (CUdevice, UInt32),
                   dev, flags)
end

@checked function cuDevicePrimaryCtxGetState(dev, flags, active)
    ccall((:cuDevicePrimaryCtxGetState, libcuda()), CUresult,
                   (CUdevice, Ptr{UInt32}, Ptr{Cint}),
                   dev, flags, active)
end

@checked function cuDevicePrimaryCtxReset_v2(dev)
    ccall((:cuDevicePrimaryCtxReset_v2, libcuda()), CUresult,
                   (CUdevice,),
                   dev)
end

@checked function cuCtxCreate_v2(pctx, flags, dev)
    ccall((:cuCtxCreate_v2, libcuda()), CUresult,
                   (Ptr{CUcontext}, UInt32, CUdevice),
                   pctx, flags, dev)
end

@checked function cuCtxDestroy_v2(ctx)
    ccall((:cuCtxDestroy_v2, libcuda()), CUresult,
                   (CUcontext,),
                   ctx)
end

@checked function cuCtxPushCurrent_v2(ctx)
    ccall((:cuCtxPushCurrent_v2, libcuda()), CUresult,
                   (CUcontext,),
                   ctx)
end

@checked function cuCtxPopCurrent_v2(pctx)
    ccall((:cuCtxPopCurrent_v2, libcuda()), CUresult,
                   (Ptr{CUcontext},),
                   pctx)
end

@checked function cuCtxSetCurrent(ctx)
    ccall((:cuCtxSetCurrent, libcuda()), CUresult,
                   (CUcontext,),
                   ctx)
end

@checked function cuCtxGetCurrent(pctx)
    ccall((:cuCtxGetCurrent, libcuda()), CUresult,
                   (Ptr{CUcontext},),
                   pctx)
end

@checked function cuCtxGetDevice(device)
    ccall((:cuCtxGetDevice, libcuda()), CUresult,
                   (Ptr{CUdevice},),
                   device)
end

@checked function cuCtxGetFlags(flags)
    initialize_api()
    ccall((:cuCtxGetFlags, libcuda()), CUresult,
                   (Ptr{UInt32},),
                   flags)
end

@checked function cuCtxSynchronize()
    initialize_api()
    ccall((:cuCtxSynchronize, libcuda()), CUresult, ())
end

@checked function cuCtxSetLimit(limit, value)
    initialize_api()
    ccall((:cuCtxSetLimit, libcuda()), CUresult,
                   (CUlimit, Csize_t),
                   limit, value)
end

@checked function cuCtxGetLimit(pvalue, limit)
    initialize_api()
    ccall((:cuCtxGetLimit, libcuda()), CUresult,
                   (Ptr{Csize_t}, CUlimit),
                   pvalue, limit)
end

@checked function cuCtxGetCacheConfig(pconfig)
    initialize_api()
    ccall((:cuCtxGetCacheConfig, libcuda()), CUresult,
                   (Ptr{CUfunc_cache},),
                   pconfig)
end

@checked function cuCtxSetCacheConfig(config)
    initialize_api()
    ccall((:cuCtxSetCacheConfig, libcuda()), CUresult,
                   (CUfunc_cache,),
                   config)
end

@checked function cuCtxGetSharedMemConfig(pConfig)
    initialize_api()
    ccall((:cuCtxGetSharedMemConfig, libcuda()), CUresult,
                   (Ptr{CUsharedconfig},),
                   pConfig)
end

@checked function cuCtxSetSharedMemConfig(config)
    initialize_api()
    ccall((:cuCtxSetSharedMemConfig, libcuda()), CUresult,
                   (CUsharedconfig,),
                   config)
end

@checked function cuCtxGetApiVersion(ctx, version)
    initialize_api()
    ccall((:cuCtxGetApiVersion, libcuda()), CUresult,
                   (CUcontext, Ptr{UInt32}),
                   ctx, version)
end

@checked function cuCtxGetStreamPriorityRange(leastPriority, greatestPriority)
    initialize_api()
    ccall((:cuCtxGetStreamPriorityRange, libcuda()), CUresult,
                   (Ptr{Cint}, Ptr{Cint}),
                   leastPriority, greatestPriority)
end

@checked function cuCtxResetPersistingL2Cache()
    initialize_api()
    ccall((:cuCtxResetPersistingL2Cache, libcuda()), CUresult, ())
end

@checked function cuCtxAttach(pctx, flags)
    initialize_api()
    ccall((:cuCtxAttach, libcuda()), CUresult,
                   (Ptr{CUcontext}, UInt32),
                   pctx, flags)
end

@checked function cuCtxDetach(ctx)
    initialize_api()
    ccall((:cuCtxDetach, libcuda()), CUresult,
                   (CUcontext,),
                   ctx)
end

@checked function cuModuleLoad(_module, fname)
    initialize_api()
    ccall((:cuModuleLoad, libcuda()), CUresult,
                   (Ptr{CUmodule}, Cstring),
                   _module, fname)
end

@checked function cuModuleLoadData(_module, image)
    initialize_api()
    ccall((:cuModuleLoadData, libcuda()), CUresult,
                   (Ptr{CUmodule}, Ptr{Cvoid}),
                   _module, image)
end

@checked function cuModuleLoadDataEx(_module, image, numOptions, options, optionValues)
    initialize_api()
    ccall((:cuModuleLoadDataEx, libcuda()), CUresult,
                   (Ptr{CUmodule}, Ptr{Cvoid}, UInt32, Ptr{CUjit_option}, Ptr{Ptr{Cvoid}}),
                   _module, image, numOptions, options, optionValues)
end

@checked function cuModuleLoadFatBinary(_module, fatCubin)
    initialize_api()
    ccall((:cuModuleLoadFatBinary, libcuda()), CUresult,
                   (Ptr{CUmodule}, Ptr{Cvoid}),
                   _module, fatCubin)
end

@checked function cuModuleUnload(hmod)
    initialize_api()
    ccall((:cuModuleUnload, libcuda()), CUresult,
                   (CUmodule,),
                   hmod)
end

@checked function cuModuleGetFunction(hfunc, hmod, name)
    initialize_api()
    ccall((:cuModuleGetFunction, libcuda()), CUresult,
                   (Ptr{CUfunction}, CUmodule, Cstring),
                   hfunc, hmod, name)
end

@checked function cuModuleGetGlobal_v2(dptr, bytes, hmod, name)
    initialize_api()
    ccall((:cuModuleGetGlobal_v2, libcuda()), CUresult,
                   (Ptr{CUdeviceptr}, Ptr{Csize_t}, CUmodule, Cstring),
                   dptr, bytes, hmod, name)
end

@checked function cuModuleGetTexRef(pTexRef, hmod, name)
    initialize_api()
    ccall((:cuModuleGetTexRef, libcuda()), CUresult,
                   (Ptr{CUtexref}, CUmodule, Cstring),
                   pTexRef, hmod, name)
end

@checked function cuModuleGetSurfRef(pSurfRef, hmod, name)
    initialize_api()
    ccall((:cuModuleGetSurfRef, libcuda()), CUresult,
                   (Ptr{CUsurfref}, CUmodule, Cstring),
                   pSurfRef, hmod, name)
end

@checked function cuLinkCreate_v2(numOptions, options, optionValues, stateOut)
    initialize_api()
    ccall((:cuLinkCreate_v2, libcuda()), CUresult,
                   (UInt32, Ptr{CUjit_option}, Ptr{Ptr{Cvoid}}, Ptr{CUlinkState}),
                   numOptions, options, optionValues, stateOut)
end

@checked function cuLinkAddData_v2(state, type, data, size, name, numOptions, options,
                                   optionValues)
    initialize_api()
    ccall((:cuLinkAddData_v2, libcuda()), CUresult,
                   (CUlinkState, CUjitInputType, Ptr{Cvoid}, Csize_t, Cstring, UInt32,
                    Ptr{CUjit_option}, Ptr{Ptr{Cvoid}}),
                   state, type, data, size, name, numOptions, options, optionValues)
end

@checked function cuLinkAddFile_v2(state, type, path, numOptions, options, optionValues)
    initialize_api()
    ccall((:cuLinkAddFile_v2, libcuda()), CUresult,
                   (CUlinkState, CUjitInputType, Cstring, UInt32, Ptr{CUjit_option},
                    Ptr{Ptr{Cvoid}}),
                   state, type, path, numOptions, options, optionValues)
end

@checked function cuLinkComplete(state, cubinOut, sizeOut)
    initialize_api()
    ccall((:cuLinkComplete, libcuda()), CUresult,
                   (CUlinkState, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}),
                   state, cubinOut, sizeOut)
end

@checked function cuLinkDestroy(state)
    initialize_api()
    ccall((:cuLinkDestroy, libcuda()), CUresult,
                   (CUlinkState,),
                   state)
end

@checked function cuMemGetInfo_v2(free, total)
    initialize_api()
    ccall((:cuMemGetInfo_v2, libcuda()), CUresult,
                   (Ptr{Csize_t}, Ptr{Csize_t}),
                   free, total)
end

@checked function cuMemAlloc_v2(dptr, bytesize)
    initialize_api()
    ccall((:cuMemAlloc_v2, libcuda()), CUresult,
                   (Ptr{CUdeviceptr}, Csize_t),
                   dptr, bytesize)
end

@checked function cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes)
    initialize_api()
    ccall((:cuMemAllocPitch_v2, libcuda()), CUresult,
                   (Ptr{CUdeviceptr}, Ptr{Csize_t}, Csize_t, Csize_t, UInt32),
                   dptr, pPitch, WidthInBytes, Height, ElementSizeBytes)
end

@checked function cuMemFree_v2(dptr)
    initialize_api()
    ccall((:cuMemFree_v2, libcuda()), CUresult,
                   (CUdeviceptr,),
                   dptr)
end

@checked function cuMemGetAddressRange_v2(pbase, psize, dptr)
    initialize_api()
    ccall((:cuMemGetAddressRange_v2, libcuda()), CUresult,
                   (Ptr{CUdeviceptr}, Ptr{Csize_t}, CUdeviceptr),
                   pbase, psize, dptr)
end

@checked function cuMemAllocHost_v2(pp, bytesize)
    initialize_api()
    ccall((:cuMemAllocHost_v2, libcuda()), CUresult,
                   (Ptr{Ptr{Cvoid}}, Csize_t),
                   pp, bytesize)
end

@checked function cuMemFreeHost(p)
    initialize_api()
    ccall((:cuMemFreeHost, libcuda()), CUresult,
                   (Ptr{Cvoid},),
                   p)
end

@checked function cuMemHostAlloc(pp, bytesize, Flags)
    initialize_api()
    ccall((:cuMemHostAlloc, libcuda()), CUresult,
                   (Ptr{Ptr{Cvoid}}, Csize_t, UInt32),
                   pp, bytesize, Flags)
end

@checked function cuMemHostGetDevicePointer_v2(pdptr, p, Flags)
    initialize_api()
    ccall((:cuMemHostGetDevicePointer_v2, libcuda()), CUresult,
                   (Ptr{CUdeviceptr}, Ptr{Cvoid}, UInt32),
                   pdptr, p, Flags)
end

@checked function cuMemHostGetFlags(pFlags, p)
    initialize_api()
    ccall((:cuMemHostGetFlags, libcuda()), CUresult,
                   (Ptr{UInt32}, Ptr{Cvoid}),
                   pFlags, p)
end

@checked function cuMemAllocManaged(dptr, bytesize, flags)
    initialize_api()
    ccall((:cuMemAllocManaged, libcuda()), CUresult,
                   (Ptr{CUdeviceptr}, Csize_t, UInt32),
                   dptr, bytesize, flags)
end

@checked function cuDeviceGetByPCIBusId(dev, pciBusId)
    initialize_api()
    ccall((:cuDeviceGetByPCIBusId, libcuda()), CUresult,
                   (Ptr{CUdevice}, Cstring),
                   dev, pciBusId)
end

@checked function cuDeviceGetPCIBusId(pciBusId, len, dev)
    initialize_api()
    ccall((:cuDeviceGetPCIBusId, libcuda()), CUresult,
                   (Cstring, Cint, CUdevice),
                   pciBusId, len, dev)
end

@checked function cuIpcGetEventHandle(pHandle, event)
    initialize_api()
    ccall((:cuIpcGetEventHandle, libcuda()), CUresult,
                   (Ptr{CUipcEventHandle}, CUevent),
                   pHandle, event)
end

@checked function cuIpcOpenEventHandle(phEvent, handle)
    initialize_api()
    ccall((:cuIpcOpenEventHandle, libcuda()), CUresult,
                   (Ptr{CUevent}, CUipcEventHandle),
                   phEvent, handle)
end

@checked function cuIpcGetMemHandle(pHandle, dptr)
    initialize_api()
    ccall((:cuIpcGetMemHandle, libcuda()), CUresult,
                   (Ptr{CUipcMemHandle}, CUdeviceptr),
                   pHandle, dptr)
end

@checked function cuIpcOpenMemHandle(pdptr, handle, Flags)
    initialize_api()
    ccall((:cuIpcOpenMemHandle, libcuda()), CUresult,
                   (Ptr{CUdeviceptr}, CUipcMemHandle, UInt32),
                   pdptr, handle, Flags)
end

@checked function cuIpcCloseMemHandle(dptr)
    initialize_api()
    ccall((:cuIpcCloseMemHandle, libcuda()), CUresult,
                   (CUdeviceptr,),
                   dptr)
end

@checked function cuMemHostRegister_v2(p, bytesize, Flags)
    initialize_api()
    ccall((:cuMemHostRegister_v2, libcuda()), CUresult,
                   (Ptr{Cvoid}, Csize_t, UInt32),
                   p, bytesize, Flags)
end

@checked function cuMemHostUnregister(p)
    initialize_api()
    ccall((:cuMemHostUnregister, libcuda()), CUresult,
                   (Ptr{Cvoid},),
                   p)
end

@checked function cuMemcpy(dst, src, ByteCount)
    initialize_api()
    ccall((:cuMemcpy_ptds, libcuda()), CUresult,
                   (CUdeviceptr, CUdeviceptr, Csize_t),
                   dst, src, ByteCount)
end

@checked function cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount)
    initialize_api()
    ccall((:cuMemcpyPeer_ptds, libcuda()), CUresult,
                   (CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, Csize_t),
                   dstDevice, dstContext, srcDevice, srcContext, ByteCount)
end

@checked function cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount)
    initialize_api()
    ccall((:cuMemcpyHtoD_v2_ptds, libcuda()), CUresult,
                   (CUdeviceptr, Ptr{Cvoid}, Csize_t),
                   dstDevice, srcHost, ByteCount)
end

@checked function cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount)
    initialize_api()
    ccall((:cuMemcpyDtoH_v2_ptds, libcuda()), CUresult,
                   (Ptr{Cvoid}, CUdeviceptr, Csize_t),
                   dstHost, srcDevice, ByteCount)
end

@checked function cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount)
    initialize_api()
    ccall((:cuMemcpyDtoD_v2_ptds, libcuda()), CUresult,
                   (CUdeviceptr, CUdeviceptr, Csize_t),
                   dstDevice, srcDevice, ByteCount)
end

@checked function cuMemcpyDtoA_v2(dstArray, dstOffset, srcDevice, ByteCount)
    initialize_api()
    ccall((:cuMemcpyDtoA_v2_ptds, libcuda()), CUresult,
                   (CUarray, Csize_t, CUdeviceptr, Csize_t),
                   dstArray, dstOffset, srcDevice, ByteCount)
end

@checked function cuMemcpyAtoD_v2(dstDevice, srcArray, srcOffset, ByteCount)
    initialize_api()
    ccall((:cuMemcpyAtoD_v2_ptds, libcuda()), CUresult,
                   (CUdeviceptr, CUarray, Csize_t, Csize_t),
                   dstDevice, srcArray, srcOffset, ByteCount)
end

@checked function cuMemcpyHtoA_v2(dstArray, dstOffset, srcHost, ByteCount)
    initialize_api()
    ccall((:cuMemcpyHtoA_v2_ptds, libcuda()), CUresult,
                   (CUarray, Csize_t, Ptr{Cvoid}, Csize_t),
                   dstArray, dstOffset, srcHost, ByteCount)
end

@checked function cuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount)
    initialize_api()
    ccall((:cuMemcpyAtoH_v2_ptds, libcuda()), CUresult,
                   (Ptr{Cvoid}, CUarray, Csize_t, Csize_t),
                   dstHost, srcArray, srcOffset, ByteCount)
end

@checked function cuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount)
    initialize_api()
    ccall((:cuMemcpyAtoA_v2_ptds, libcuda()), CUresult,
                   (CUarray, Csize_t, CUarray, Csize_t, Csize_t),
                   dstArray, dstOffset, srcArray, srcOffset, ByteCount)
end

@checked function cuMemcpy2D_v2(pCopy)
    initialize_api()
    ccall((:cuMemcpy2D_v2_ptds, libcuda()), CUresult,
                   (Ptr{CUDA_MEMCPY2D},),
                   pCopy)
end

@checked function cuMemcpy2DUnaligned_v2(pCopy)
    initialize_api()
    ccall((:cuMemcpy2DUnaligned_v2_ptds, libcuda()), CUresult,
                   (Ptr{CUDA_MEMCPY2D},),
                   pCopy)
end

@checked function cuMemcpy3D_v2(pCopy)
    initialize_api()
    ccall((:cuMemcpy3D_v2_ptds, libcuda()), CUresult,
                   (Ptr{CUDA_MEMCPY3D},),
                   pCopy)
end

@checked function cuMemcpy3DPeer(pCopy)
    initialize_api()
    ccall((:cuMemcpy3DPeer_ptds, libcuda()), CUresult,
                   (Ptr{CUDA_MEMCPY3D_PEER},),
                   pCopy)
end

@checked function cuMemcpyAsync(dst, src, ByteCount, hStream)
    initialize_api()
    ccall((:cuMemcpyAsync_ptsz, libcuda()), CUresult,
                   (CUdeviceptr, CUdeviceptr, Csize_t, CUstream),
                   dst, src, ByteCount, hStream)
end

@checked function cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext,
                                    ByteCount, hStream)
    initialize_api()
    ccall((:cuMemcpyPeerAsync_ptsz, libcuda()), CUresult,
                   (CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, Csize_t, CUstream),
                   dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream)
end

@checked function cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream)
    initialize_api()
    ccall((:cuMemcpyHtoDAsync_v2_ptsz, libcuda()), CUresult,
                   (CUdeviceptr, Ptr{Cvoid}, Csize_t, CUstream),
                   dstDevice, srcHost, ByteCount, hStream)
end

@checked function cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream)
    initialize_api()
    ccall((:cuMemcpyDtoHAsync_v2_ptsz, libcuda()), CUresult,
                   (Ptr{Cvoid}, CUdeviceptr, Csize_t, CUstream),
                   dstHost, srcDevice, ByteCount, hStream)
end

@checked function cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream)
    initialize_api()
    ccall((:cuMemcpyDtoDAsync_v2_ptsz, libcuda()), CUresult,
                   (CUdeviceptr, CUdeviceptr, Csize_t, CUstream),
                   dstDevice, srcDevice, ByteCount, hStream)
end

@checked function cuMemcpyHtoAAsync_v2(dstArray, dstOffset, srcHost, ByteCount, hStream)
    initialize_api()
    ccall((:cuMemcpyHtoAAsync_v2_ptsz, libcuda()), CUresult,
                   (CUarray, Csize_t, Ptr{Cvoid}, Csize_t, CUstream),
                   dstArray, dstOffset, srcHost, ByteCount, hStream)
end

@checked function cuMemcpyAtoHAsync_v2(dstHost, srcArray, srcOffset, ByteCount, hStream)
    initialize_api()
    ccall((:cuMemcpyAtoHAsync_v2_ptsz, libcuda()), CUresult,
                   (Ptr{Cvoid}, CUarray, Csize_t, Csize_t, CUstream),
                   dstHost, srcArray, srcOffset, ByteCount, hStream)
end

@checked function cuMemcpy2DAsync_v2(pCopy, hStream)
    initialize_api()
    ccall((:cuMemcpy2DAsync_v2_ptsz, libcuda()), CUresult,
                   (Ptr{CUDA_MEMCPY2D}, CUstream),
                   pCopy, hStream)
end

@checked function cuMemcpy3DAsync_v2(pCopy, hStream)
    initialize_api()
    ccall((:cuMemcpy3DAsync_v2_ptsz, libcuda()), CUresult,
                   (Ptr{CUDA_MEMCPY3D}, CUstream),
                   pCopy, hStream)
end

@checked function cuMemcpy3DPeerAsync(pCopy, hStream)
    initialize_api()
    ccall((:cuMemcpy3DPeerAsync_ptsz, libcuda()), CUresult,
                   (Ptr{CUDA_MEMCPY3D_PEER}, CUstream),
                   pCopy, hStream)
end

@checked function cuMemsetD8_v2(dstDevice, uc, N)
    initialize_api()
    ccall((:cuMemsetD8_v2_ptds, libcuda()), CUresult,
                   (CUdeviceptr, Cuchar, Csize_t),
                   dstDevice, uc, N)
end

@checked function cuMemsetD16_v2(dstDevice, us, N)
    initialize_api()
    ccall((:cuMemsetD16_v2_ptds, libcuda()), CUresult,
                   (CUdeviceptr, UInt16, Csize_t),
                   dstDevice, us, N)
end

@checked function cuMemsetD32_v2(dstDevice, ui, N)
    initialize_api()
    ccall((:cuMemsetD32_v2_ptds, libcuda()), CUresult,
                   (CUdeviceptr, UInt32, Csize_t),
                   dstDevice, ui, N)
end

@checked function cuMemsetD2D8_v2(dstDevice, dstPitch, uc, Width, Height)
    initialize_api()
    ccall((:cuMemsetD2D8_v2_ptds, libcuda()), CUresult,
                   (CUdeviceptr, Csize_t, Cuchar, Csize_t, Csize_t),
                   dstDevice, dstPitch, uc, Width, Height)
end

@checked function cuMemsetD2D16_v2(dstDevice, dstPitch, us, Width, Height)
    initialize_api()
    ccall((:cuMemsetD2D16_v2_ptds, libcuda()), CUresult,
                   (CUdeviceptr, Csize_t, UInt16, Csize_t, Csize_t),
                   dstDevice, dstPitch, us, Width, Height)
end

@checked function cuMemsetD2D32_v2(dstDevice, dstPitch, ui, Width, Height)
    initialize_api()
    ccall((:cuMemsetD2D32_v2_ptds, libcuda()), CUresult,
                   (CUdeviceptr, Csize_t, UInt32, Csize_t, Csize_t),
                   dstDevice, dstPitch, ui, Width, Height)
end

@checked function cuMemsetD8Async(dstDevice, uc, N, hStream)
    initialize_api()
    ccall((:cuMemsetD8Async_ptsz, libcuda()), CUresult,
                   (CUdeviceptr, Cuchar, Csize_t, CUstream),
                   dstDevice, uc, N, hStream)
end

@checked function cuMemsetD16Async(dstDevice, us, N, hStream)
    initialize_api()
    ccall((:cuMemsetD16Async_ptsz, libcuda()), CUresult,
                   (CUdeviceptr, UInt16, Csize_t, CUstream),
                   dstDevice, us, N, hStream)
end

@checked function cuMemsetD32Async(dstDevice, ui, N, hStream)
    initialize_api()
    ccall((:cuMemsetD32Async_ptsz, libcuda()), CUresult,
                   (CUdeviceptr, UInt32, Csize_t, CUstream),
                   dstDevice, ui, N, hStream)
end

@checked function cuMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream)
    initialize_api()
    ccall((:cuMemsetD2D8Async_ptsz, libcuda()), CUresult,
                   (CUdeviceptr, Csize_t, Cuchar, Csize_t, Csize_t, CUstream),
                   dstDevice, dstPitch, uc, Width, Height, hStream)
end

@checked function cuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream)
    initialize_api()
    ccall((:cuMemsetD2D16Async_ptsz, libcuda()), CUresult,
                   (CUdeviceptr, Csize_t, UInt16, Csize_t, Csize_t, CUstream),
                   dstDevice, dstPitch, us, Width, Height, hStream)
end

@checked function cuMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream)
    initialize_api()
    ccall((:cuMemsetD2D32Async_ptsz, libcuda()), CUresult,
                   (CUdeviceptr, Csize_t, UInt32, Csize_t, Csize_t, CUstream),
                   dstDevice, dstPitch, ui, Width, Height, hStream)
end

@checked function cuArrayCreate_v2(pHandle, pAllocateArray)
    initialize_api()
    ccall((:cuArrayCreate_v2, libcuda()), CUresult,
                   (Ptr{CUarray}, Ptr{CUDA_ARRAY_DESCRIPTOR}),
                   pHandle, pAllocateArray)
end

@checked function cuArrayGetDescriptor_v2(pArrayDescriptor, hArray)
    initialize_api()
    ccall((:cuArrayGetDescriptor_v2, libcuda()), CUresult,
                   (Ptr{CUDA_ARRAY_DESCRIPTOR}, CUarray),
                   pArrayDescriptor, hArray)
end

@checked function cuArrayDestroy(hArray)
    initialize_api()
    ccall((:cuArrayDestroy, libcuda()), CUresult,
                   (CUarray,),
                   hArray)
end

@checked function cuArray3DCreate_v2(pHandle, pAllocateArray)
    initialize_api()
    ccall((:cuArray3DCreate_v2, libcuda()), CUresult,
                   (Ptr{CUarray}, Ptr{CUDA_ARRAY3D_DESCRIPTOR}),
                   pHandle, pAllocateArray)
end

@checked function cuArray3DGetDescriptor_v2(pArrayDescriptor, hArray)
    initialize_api()
    ccall((:cuArray3DGetDescriptor_v2, libcuda()), CUresult,
                   (Ptr{CUDA_ARRAY3D_DESCRIPTOR}, CUarray),
                   pArrayDescriptor, hArray)
end

@checked function cuMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels)
    initialize_api()
    ccall((:cuMipmappedArrayCreate, libcuda()), CUresult,
                   (Ptr{CUmipmappedArray}, Ptr{CUDA_ARRAY3D_DESCRIPTOR}, UInt32),
                   pHandle, pMipmappedArrayDesc, numMipmapLevels)
end

@checked function cuMipmappedArrayGetLevel(pLevelArray, hMipmappedArray, level)
    initialize_api()
    ccall((:cuMipmappedArrayGetLevel, libcuda()), CUresult,
                   (Ptr{CUarray}, CUmipmappedArray, UInt32),
                   pLevelArray, hMipmappedArray, level)
end

@checked function cuMipmappedArrayDestroy(hMipmappedArray)
    initialize_api()
    ccall((:cuMipmappedArrayDestroy, libcuda()), CUresult,
                   (CUmipmappedArray,),
                   hMipmappedArray)
end

@checked function cuMemAddressReserve(ptr, size, alignment, addr, flags)
    initialize_api()
    ccall((:cuMemAddressReserve, libcuda()), CUresult,
                   (Ptr{CUdeviceptr}, Csize_t, Csize_t, CUdeviceptr, Culonglong),
                   ptr, size, alignment, addr, flags)
end

@checked function cuMemAddressFree(ptr, size)
    initialize_api()
    ccall((:cuMemAddressFree, libcuda()), CUresult,
                   (CUdeviceptr, Csize_t),
                   ptr, size)
end

@checked function cuMemCreate(handle, size, prop, flags)
    initialize_api()
    ccall((:cuMemCreate, libcuda()), CUresult,
                   (Ptr{CUmemGenericAllocationHandle}, Csize_t, Ptr{CUmemAllocationProp},
                    Culonglong),
                   handle, size, prop, flags)
end

@checked function cuMemRelease(handle)
    initialize_api()
    ccall((:cuMemRelease, libcuda()), CUresult,
                   (CUmemGenericAllocationHandle,),
                   handle)
end

@checked function cuMemMap(ptr, size, offset, handle, flags)
    initialize_api()
    ccall((:cuMemMap, libcuda()), CUresult,
                   (CUdeviceptr, Csize_t, Csize_t, CUmemGenericAllocationHandle,
                    Culonglong),
                   ptr, size, offset, handle, flags)
end

@checked function cuMemUnmap(ptr, size)
    initialize_api()
    ccall((:cuMemUnmap, libcuda()), CUresult,
                   (CUdeviceptr, Csize_t),
                   ptr, size)
end

@checked function cuMemSetAccess(ptr, size, desc, count)
    initialize_api()
    ccall((:cuMemSetAccess, libcuda()), CUresult,
                   (CUdeviceptr, Csize_t, Ptr{CUmemAccessDesc}, Csize_t),
                   ptr, size, desc, count)
end

@checked function cuMemGetAccess(flags, location, ptr)
    initialize_api()
    ccall((:cuMemGetAccess, libcuda()), CUresult,
                   (Ptr{Culonglong}, Ptr{CUmemLocation}, CUdeviceptr),
                   flags, location, ptr)
end

@checked function cuMemExportToShareableHandle(shareableHandle, handle, handleType, flags)
    initialize_api()
    ccall((:cuMemExportToShareableHandle, libcuda()), CUresult,
                   (Ptr{Cvoid}, CUmemGenericAllocationHandle, CUmemAllocationHandleType,
                    Culonglong),
                   shareableHandle, handle, handleType, flags)
end

@checked function cuMemImportFromShareableHandle(handle, osHandle, shHandleType)
    initialize_api()
    ccall((:cuMemImportFromShareableHandle, libcuda()), CUresult,
                   (Ptr{CUmemGenericAllocationHandle}, Ptr{Cvoid},
                    CUmemAllocationHandleType),
                   handle, osHandle, shHandleType)
end

@checked function cuMemGetAllocationGranularity(granularity, prop, option)
    initialize_api()
    ccall((:cuMemGetAllocationGranularity, libcuda()), CUresult,
                   (Ptr{Csize_t}, Ptr{CUmemAllocationProp},
                    CUmemAllocationGranularity_flags),
                   granularity, prop, option)
end

@checked function cuMemGetAllocationPropertiesFromHandle(prop, handle)
    initialize_api()
    ccall((:cuMemGetAllocationPropertiesFromHandle, libcuda()), CUresult,
                   (Ptr{CUmemAllocationProp}, CUmemGenericAllocationHandle),
                   prop, handle)
end

@checked function cuMemRetainAllocationHandle(handle, addr)
    initialize_api()
    ccall((:cuMemRetainAllocationHandle, libcuda()), CUresult,
                   (Ptr{CUmemGenericAllocationHandle}, Ptr{Cvoid}),
                   handle, addr)
end

@checked function cuPointerGetAttribute(data, attribute, ptr)
    initialize_api()
    ccall((:cuPointerGetAttribute, libcuda()), CUresult,
                   (Ptr{Cvoid}, CUpointer_attribute, CUdeviceptr),
                   data, attribute, ptr)
end

@checked function cuMemPrefetchAsync(devPtr, count, dstDevice, hStream)
    initialize_api()
    ccall((:cuMemPrefetchAsync_ptsz, libcuda()), CUresult,
                   (CUdeviceptr, Csize_t, CUdevice, CUstream),
                   devPtr, count, dstDevice, hStream)
end

@checked function cuMemAdvise(devPtr, count, advice, device)
    initialize_api()
    ccall((:cuMemAdvise, libcuda()), CUresult,
                   (CUdeviceptr, Csize_t, CUmem_advise, CUdevice),
                   devPtr, count, advice, device)
end

@checked function cuMemRangeGetAttribute(data, dataSize, attribute, devPtr, count)
    initialize_api()
    ccall((:cuMemRangeGetAttribute, libcuda()), CUresult,
                   (Ptr{Cvoid}, Csize_t, CUmem_range_attribute, CUdeviceptr, Csize_t),
                   data, dataSize, attribute, devPtr, count)
end

@checked function cuMemRangeGetAttributes(data, dataSizes, attributes, numAttributes,
                                          devPtr, count)
    initialize_api()
    ccall((:cuMemRangeGetAttributes, libcuda()), CUresult,
                   (Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{CUmem_range_attribute}, Csize_t,
                    CUdeviceptr, Csize_t),
                   data, dataSizes, attributes, numAttributes, devPtr, count)
end

@checked function cuPointerSetAttribute(value, attribute, ptr)
    initialize_api()
    ccall((:cuPointerSetAttribute, libcuda()), CUresult,
                   (Ptr{Cvoid}, CUpointer_attribute, CUdeviceptr),
                   value, attribute, ptr)
end

@checked function cuPointerGetAttributes(numAttributes, attributes, data, ptr)
    initialize_api()
    ccall((:cuPointerGetAttributes, libcuda()), CUresult,
                   (UInt32, Ptr{CUpointer_attribute}, Ptr{Ptr{Cvoid}}, CUdeviceptr),
                   numAttributes, attributes, data, ptr)
end

@checked function cuStreamCreate(phStream, Flags)
    initialize_api()
    ccall((:cuStreamCreate, libcuda()), CUresult,
                   (Ptr{CUstream}, UInt32),
                   phStream, Flags)
end

@checked function cuStreamCreateWithPriority(phStream, flags, priority)
    initialize_api()
    ccall((:cuStreamCreateWithPriority, libcuda()), CUresult,
                   (Ptr{CUstream}, UInt32, Cint),
                   phStream, flags, priority)
end

@checked function cuStreamGetPriority(hStream, priority)
    initialize_api()
    ccall((:cuStreamGetPriority_ptsz, libcuda()), CUresult,
                   (CUstream, Ptr{Cint}),
                   hStream, priority)
end

@checked function cuStreamGetFlags(hStream, flags)
    initialize_api()
    ccall((:cuStreamGetFlags_ptsz, libcuda()), CUresult,
                   (CUstream, Ptr{UInt32}),
                   hStream, flags)
end

@checked function cuStreamGetCtx(hStream, pctx)
    initialize_api()
    ccall((:cuStreamGetCtx_ptsz, libcuda()), CUresult,
                   (CUstream, Ptr{CUcontext}),
                   hStream, pctx)
end

@checked function cuStreamWaitEvent(hStream, hEvent, Flags)
    initialize_api()
    ccall((:cuStreamWaitEvent_ptsz, libcuda()), CUresult,
                   (CUstream, CUevent, UInt32),
                   hStream, hEvent, Flags)
end

@checked function cuStreamAddCallback(hStream, callback, userData, flags)
    initialize_api()
    ccall((:cuStreamAddCallback_ptsz, libcuda()), CUresult,
                   (CUstream, CUstreamCallback, Ptr{Cvoid}, UInt32),
                   hStream, callback, userData, flags)
end

@checked function cuStreamBeginCapture_v2(hStream, mode)
    initialize_api()
    ccall((:cuStreamBeginCapture_v2_ptsz, libcuda()), CUresult,
                   (CUstream, CUstreamCaptureMode),
                   hStream, mode)
end

@checked function cuThreadExchangeStreamCaptureMode(mode)
    initialize_api()
    ccall((:cuThreadExchangeStreamCaptureMode, libcuda()), CUresult,
                   (Ptr{CUstreamCaptureMode},),
                   mode)
end

@checked function cuStreamEndCapture(hStream, phGraph)
    initialize_api()
    ccall((:cuStreamEndCapture_ptsz, libcuda()), CUresult,
                   (CUstream, Ptr{CUgraph}),
                   hStream, phGraph)
end

@checked function cuStreamIsCapturing(hStream, captureStatus)
    initialize_api()
    ccall((:cuStreamIsCapturing_ptsz, libcuda()), CUresult,
                   (CUstream, Ptr{CUstreamCaptureStatus}),
                   hStream, captureStatus)
end

@checked function cuStreamGetCaptureInfo(hStream, captureStatus, id)
    initialize_api()
    ccall((:cuStreamGetCaptureInfo_ptsz, libcuda()), CUresult,
                   (CUstream, Ptr{CUstreamCaptureStatus}, Ptr{cuuint64_t}),
                   hStream, captureStatus, id)
end

@checked function cuStreamAttachMemAsync(hStream, dptr, length, flags)
    initialize_api()
    ccall((:cuStreamAttachMemAsync_ptsz, libcuda()), CUresult,
                   (CUstream, CUdeviceptr, Csize_t, UInt32),
                   hStream, dptr, length, flags)
end

@checked function cuStreamQuery(hStream)
    initialize_api()
    ccall((:cuStreamQuery_ptsz, libcuda()), CUresult,
                   (CUstream,),
                   hStream)
end

@checked function cuStreamSynchronize(hStream)
    initialize_api()
    ccall((:cuStreamSynchronize_ptsz, libcuda()), CUresult,
                   (CUstream,),
                   hStream)
end

@checked function cuStreamDestroy_v2(hStream)
    initialize_api()
    ccall((:cuStreamDestroy_v2, libcuda()), CUresult,
                   (CUstream,),
                   hStream)
end

@checked function cuStreamCopyAttributes(dst, src)
    initialize_api()
    ccall((:cuStreamCopyAttributes_ptsz, libcuda()), CUresult,
                   (CUstream, CUstream),
                   dst, src)
end

@checked function cuStreamGetAttribute(hStream, attr, value_out)
    initialize_api()
    ccall((:cuStreamGetAttribute_ptsz, libcuda()), CUresult,
                   (CUstream, CUstreamAttrID, Ptr{CUstreamAttrValue}),
                   hStream, attr, value_out)
end

@checked function cuStreamSetAttribute(hStream, attr, value)
    initialize_api()
    ccall((:cuStreamSetAttribute_ptsz, libcuda()), CUresult,
                   (CUstream, CUstreamAttrID, Ptr{CUstreamAttrValue}),
                   hStream, attr, value)
end

@checked function cuEventCreate(phEvent, Flags)
    initialize_api()
    ccall((:cuEventCreate, libcuda()), CUresult,
                   (Ptr{CUevent}, UInt32),
                   phEvent, Flags)
end

@checked function cuEventRecord(hEvent, hStream)
    initialize_api()
    ccall((:cuEventRecord_ptsz, libcuda()), CUresult,
                   (CUevent, CUstream),
                   hEvent, hStream)
end

@checked function cuEventQuery(hEvent)
    initialize_api()
    ccall((:cuEventQuery, libcuda()), CUresult,
                   (CUevent,),
                   hEvent)
end

@checked function cuEventSynchronize(hEvent)
    initialize_api()
    ccall((:cuEventSynchronize, libcuda()), CUresult,
                   (CUevent,),
                   hEvent)
end

@checked function cuEventDestroy_v2(hEvent)
    initialize_api()
    ccall((:cuEventDestroy_v2, libcuda()), CUresult,
                   (CUevent,),
                   hEvent)
end

@checked function cuEventElapsedTime(pMilliseconds, hStart, hEnd)
    initialize_api()
    ccall((:cuEventElapsedTime, libcuda()), CUresult,
                   (Ptr{Cfloat}, CUevent, CUevent),
                   pMilliseconds, hStart, hEnd)
end

@checked function cuImportExternalMemory(extMem_out, memHandleDesc)
    initialize_api()
    ccall((:cuImportExternalMemory, libcuda()), CUresult,
                   (Ptr{CUexternalMemory}, Ptr{CUDA_EXTERNAL_MEMORY_HANDLE_DESC}),
                   extMem_out, memHandleDesc)
end

@checked function cuExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc)
    initialize_api()
    ccall((:cuExternalMemoryGetMappedBuffer, libcuda()), CUresult,
                   (Ptr{CUdeviceptr}, CUexternalMemory,
                    Ptr{CUDA_EXTERNAL_MEMORY_BUFFER_DESC}),
                   devPtr, extMem, bufferDesc)
end

@checked function cuExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc)
    initialize_api()
    ccall((:cuExternalMemoryGetMappedMipmappedArray, libcuda()), CUresult,
                   (Ptr{CUmipmappedArray}, CUexternalMemory,
                    Ptr{CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC}),
                   mipmap, extMem, mipmapDesc)
end

@checked function cuDestroyExternalMemory(extMem)
    initialize_api()
    ccall((:cuDestroyExternalMemory, libcuda()), CUresult,
                   (CUexternalMemory,),
                   extMem)
end

@checked function cuImportExternalSemaphore(extSem_out, semHandleDesc)
    initialize_api()
    ccall((:cuImportExternalSemaphore, libcuda()), CUresult,
                   (Ptr{CUexternalSemaphore}, Ptr{CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC}),
                   extSem_out, semHandleDesc)
end

@checked function cuSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems,
                                                  stream)
    initialize_api()
    ccall((:cuSignalExternalSemaphoresAsync_ptsz, libcuda()), CUresult,
                   (Ptr{CUexternalSemaphore}, Ptr{CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS},
                    UInt32, CUstream),
                   extSemArray, paramsArray, numExtSems, stream)
end

@checked function cuWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)
    initialize_api()
    ccall((:cuWaitExternalSemaphoresAsync_ptsz, libcuda()), CUresult,
                   (Ptr{CUexternalSemaphore}, Ptr{CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS},
                    UInt32, CUstream),
                   extSemArray, paramsArray, numExtSems, stream)
end

@checked function cuDestroyExternalSemaphore(extSem)
    initialize_api()
    ccall((:cuDestroyExternalSemaphore, libcuda()), CUresult,
                   (CUexternalSemaphore,),
                   extSem)
end

@checked function cuStreamWaitValue32(stream, addr, value, flags)
    initialize_api()
    ccall((:cuStreamWaitValue32_ptsz, libcuda()), CUresult,
                   (CUstream, CUdeviceptr, cuuint32_t, UInt32),
                   stream, addr, value, flags)
end

@checked function cuStreamWaitValue64(stream, addr, value, flags)
    initialize_api()
    ccall((:cuStreamWaitValue64_ptsz, libcuda()), CUresult,
                   (CUstream, CUdeviceptr, cuuint64_t, UInt32),
                   stream, addr, value, flags)
end

@checked function cuStreamWriteValue32(stream, addr, value, flags)
    initialize_api()
    ccall((:cuStreamWriteValue32_ptsz, libcuda()), CUresult,
                   (CUstream, CUdeviceptr, cuuint32_t, UInt32),
                   stream, addr, value, flags)
end

@checked function cuStreamWriteValue64(stream, addr, value, flags)
    initialize_api()
    ccall((:cuStreamWriteValue64_ptsz, libcuda()), CUresult,
                   (CUstream, CUdeviceptr, cuuint64_t, UInt32),
                   stream, addr, value, flags)
end

@checked function cuFuncGetAttribute(pi, attrib, hfunc)
    initialize_api()
    ccall((:cuFuncGetAttribute, libcuda()), CUresult,
                   (Ptr{Cint}, CUfunction_attribute, CUfunction),
                   pi, attrib, hfunc)
end

@checked function cuFuncSetAttribute(hfunc, attrib, value)
    initialize_api()
    ccall((:cuFuncSetAttribute, libcuda()), CUresult,
                   (CUfunction, CUfunction_attribute, Cint),
                   hfunc, attrib, value)
end

@checked function cuFuncSetCacheConfig(hfunc, config)
    initialize_api()
    ccall((:cuFuncSetCacheConfig, libcuda()), CUresult,
                   (CUfunction, CUfunc_cache),
                   hfunc, config)
end

@checked function cuFuncSetSharedMemConfig(hfunc, config)
    initialize_api()
    ccall((:cuFuncSetSharedMemConfig, libcuda()), CUresult,
                   (CUfunction, CUsharedconfig),
                   hfunc, config)
end

@checked function cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                                 blockDimZ, sharedMemBytes, hStream, kernelParams, extra)
    initialize_api()
    ccall((:cuLaunchKernel_ptsz, libcuda()), CUresult,
                   (CUfunction, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                    CUstream, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
                   f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                   sharedMemBytes, hStream, kernelParams, extra)
end

@checked function cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                                            blockDimY, blockDimZ, sharedMemBytes, hStream,
                                            kernelParams)
    initialize_api()
    ccall((:cuLaunchCooperativeKernel_ptsz, libcuda()), CUresult,
                   (CUfunction, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                    CUstream, Ptr{Ptr{Cvoid}}),
                   f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                   sharedMemBytes, hStream, kernelParams)
end

@checked function cuLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags)
    initialize_api()
    ccall((:cuLaunchCooperativeKernelMultiDevice, libcuda()), CUresult,
                   (Ptr{CUDA_LAUNCH_PARAMS}, UInt32, UInt32),
                   launchParamsList, numDevices, flags)
end

@checked function cuLaunchHostFunc(hStream, fn, userData)
    initialize_api()
    ccall((:cuLaunchHostFunc_ptsz, libcuda()), CUresult,
                   (CUstream, CUhostFn, Ptr{Cvoid}),
                   hStream, fn, userData)
end

@checked function cuFuncSetBlockShape(hfunc, x, y, z)
    initialize_api()
    ccall((:cuFuncSetBlockShape, libcuda()), CUresult,
                   (CUfunction, Cint, Cint, Cint),
                   hfunc, x, y, z)
end

@checked function cuFuncSetSharedSize(hfunc, bytes)
    initialize_api()
    ccall((:cuFuncSetSharedSize, libcuda()), CUresult,
                   (CUfunction, UInt32),
                   hfunc, bytes)
end

@checked function cuParamSetSize(hfunc, numbytes)
    initialize_api()
    ccall((:cuParamSetSize, libcuda()), CUresult,
                   (CUfunction, UInt32),
                   hfunc, numbytes)
end

@checked function cuParamSeti(hfunc, offset, value)
    initialize_api()
    ccall((:cuParamSeti, libcuda()), CUresult,
                   (CUfunction, Cint, UInt32),
                   hfunc, offset, value)
end

@checked function cuParamSetf(hfunc, offset, value)
    initialize_api()
    ccall((:cuParamSetf, libcuda()), CUresult,
                   (CUfunction, Cint, Cfloat),
                   hfunc, offset, value)
end

@checked function cuParamSetv(hfunc, offset, ptr, numbytes)
    initialize_api()
    ccall((:cuParamSetv, libcuda()), CUresult,
                   (CUfunction, Cint, Ptr{Cvoid}, UInt32),
                   hfunc, offset, ptr, numbytes)
end

@checked function cuLaunch(f)
    initialize_api()
    ccall((:cuLaunch, libcuda()), CUresult,
                   (CUfunction,),
                   f)
end

@checked function cuLaunchGrid(f, grid_width, grid_height)
    initialize_api()
    ccall((:cuLaunchGrid, libcuda()), CUresult,
                   (CUfunction, Cint, Cint),
                   f, grid_width, grid_height)
end

@checked function cuLaunchGridAsync(f, grid_width, grid_height, hStream)
    initialize_api()
    ccall((:cuLaunchGridAsync, libcuda()), CUresult,
                   (CUfunction, Cint, Cint, CUstream),
                   f, grid_width, grid_height, hStream)
end

@checked function cuParamSetTexRef(hfunc, texunit, hTexRef)
    initialize_api()
    ccall((:cuParamSetTexRef, libcuda()), CUresult,
                   (CUfunction, Cint, CUtexref),
                   hfunc, texunit, hTexRef)
end

@checked function cuGraphCreate(phGraph, flags)
    initialize_api()
    ccall((:cuGraphCreate, libcuda()), CUresult,
                   (Ptr{CUgraph}, UInt32),
                   phGraph, flags)
end

@checked function cuGraphAddKernelNode(phGraphNode, hGraph, dependencies, numDependencies,
                                       nodeParams)
    initialize_api()
    ccall((:cuGraphAddKernelNode, libcuda()), CUresult,
                   (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t,
                    Ptr{CUDA_KERNEL_NODE_PARAMS}),
                   phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
end

@checked function cuGraphKernelNodeGetParams(hNode, nodeParams)
    initialize_api()
    ccall((:cuGraphKernelNodeGetParams, libcuda()), CUresult,
                   (CUgraphNode, Ptr{CUDA_KERNEL_NODE_PARAMS}),
                   hNode, nodeParams)
end

@checked function cuGraphKernelNodeSetParams(hNode, nodeParams)
    initialize_api()
    ccall((:cuGraphKernelNodeSetParams, libcuda()), CUresult,
                   (CUgraphNode, Ptr{CUDA_KERNEL_NODE_PARAMS}),
                   hNode, nodeParams)
end

@checked function cuGraphAddMemcpyNode(phGraphNode, hGraph, dependencies, numDependencies,
                                       copyParams, ctx)
    initialize_api()
    ccall((:cuGraphAddMemcpyNode, libcuda()), CUresult,
                   (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t,
                    Ptr{CUDA_MEMCPY3D}, CUcontext),
                   phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx)
end

@checked function cuGraphMemcpyNodeGetParams(hNode, nodeParams)
    initialize_api()
    ccall((:cuGraphMemcpyNodeGetParams, libcuda()), CUresult,
                   (CUgraphNode, Ptr{CUDA_MEMCPY3D}),
                   hNode, nodeParams)
end

@checked function cuGraphMemcpyNodeSetParams(hNode, nodeParams)
    initialize_api()
    ccall((:cuGraphMemcpyNodeSetParams, libcuda()), CUresult,
                   (CUgraphNode, Ptr{CUDA_MEMCPY3D}),
                   hNode, nodeParams)
end

@checked function cuGraphAddMemsetNode(phGraphNode, hGraph, dependencies, numDependencies,
                                       memsetParams, ctx)
    initialize_api()
    ccall((:cuGraphAddMemsetNode, libcuda()), CUresult,
                   (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t,
                    Ptr{CUDA_MEMSET_NODE_PARAMS}, CUcontext),
                   phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx)
end

@checked function cuGraphMemsetNodeGetParams(hNode, nodeParams)
    initialize_api()
    ccall((:cuGraphMemsetNodeGetParams, libcuda()), CUresult,
                   (CUgraphNode, Ptr{CUDA_MEMSET_NODE_PARAMS}),
                   hNode, nodeParams)
end

@checked function cuGraphMemsetNodeSetParams(hNode, nodeParams)
    initialize_api()
    ccall((:cuGraphMemsetNodeSetParams, libcuda()), CUresult,
                   (CUgraphNode, Ptr{CUDA_MEMSET_NODE_PARAMS}),
                   hNode, nodeParams)
end

@checked function cuGraphAddHostNode(phGraphNode, hGraph, dependencies, numDependencies,
                                     nodeParams)
    initialize_api()
    ccall((:cuGraphAddHostNode, libcuda()), CUresult,
                   (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t,
                    Ptr{CUDA_HOST_NODE_PARAMS}),
                   phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
end

@checked function cuGraphHostNodeGetParams(hNode, nodeParams)
    initialize_api()
    ccall((:cuGraphHostNodeGetParams, libcuda()), CUresult,
                   (CUgraphNode, Ptr{CUDA_HOST_NODE_PARAMS}),
                   hNode, nodeParams)
end

@checked function cuGraphHostNodeSetParams(hNode, nodeParams)
    initialize_api()
    ccall((:cuGraphHostNodeSetParams, libcuda()), CUresult,
                   (CUgraphNode, Ptr{CUDA_HOST_NODE_PARAMS}),
                   hNode, nodeParams)
end

@checked function cuGraphAddChildGraphNode(phGraphNode, hGraph, dependencies,
                                           numDependencies, childGraph)
    initialize_api()
    ccall((:cuGraphAddChildGraphNode, libcuda()), CUresult,
                   (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t, CUgraph),
                   phGraphNode, hGraph, dependencies, numDependencies, childGraph)
end

@checked function cuGraphChildGraphNodeGetGraph(hNode, phGraph)
    initialize_api()
    ccall((:cuGraphChildGraphNodeGetGraph, libcuda()), CUresult,
                   (CUgraphNode, Ptr{CUgraph}),
                   hNode, phGraph)
end

@checked function cuGraphAddEmptyNode(phGraphNode, hGraph, dependencies, numDependencies)
    initialize_api()
    ccall((:cuGraphAddEmptyNode, libcuda()), CUresult,
                   (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t),
                   phGraphNode, hGraph, dependencies, numDependencies)
end

@checked function cuGraphClone(phGraphClone, originalGraph)
    initialize_api()
    ccall((:cuGraphClone, libcuda()), CUresult,
                   (Ptr{CUgraph}, CUgraph),
                   phGraphClone, originalGraph)
end

@checked function cuGraphNodeFindInClone(phNode, hOriginalNode, hClonedGraph)
    initialize_api()
    ccall((:cuGraphNodeFindInClone, libcuda()), CUresult,
                   (Ptr{CUgraphNode}, CUgraphNode, CUgraph),
                   phNode, hOriginalNode, hClonedGraph)
end

@checked function cuGraphNodeGetType(hNode, type)
    initialize_api()
    ccall((:cuGraphNodeGetType, libcuda()), CUresult,
                   (CUgraphNode, Ptr{CUgraphNodeType}),
                   hNode, type)
end

@checked function cuGraphGetNodes(hGraph, nodes, numNodes)
    initialize_api()
    ccall((:cuGraphGetNodes, libcuda()), CUresult,
                   (CUgraph, Ptr{CUgraphNode}, Ptr{Csize_t}),
                   hGraph, nodes, numNodes)
end

@checked function cuGraphGetRootNodes(hGraph, rootNodes, numRootNodes)
    initialize_api()
    ccall((:cuGraphGetRootNodes, libcuda()), CUresult,
                   (CUgraph, Ptr{CUgraphNode}, Ptr{Csize_t}),
                   hGraph, rootNodes, numRootNodes)
end

@checked function cuGraphGetEdges(hGraph, from, to, numEdges)
    initialize_api()
    ccall((:cuGraphGetEdges, libcuda()), CUresult,
                   (CUgraph, Ptr{CUgraphNode}, Ptr{CUgraphNode}, Ptr{Csize_t}),
                   hGraph, from, to, numEdges)
end

@checked function cuGraphNodeGetDependencies(hNode, dependencies, numDependencies)
    initialize_api()
    ccall((:cuGraphNodeGetDependencies, libcuda()), CUresult,
                   (CUgraphNode, Ptr{CUgraphNode}, Ptr{Csize_t}),
                   hNode, dependencies, numDependencies)
end

@checked function cuGraphNodeGetDependentNodes(hNode, dependentNodes, numDependentNodes)
    initialize_api()
    ccall((:cuGraphNodeGetDependentNodes, libcuda()), CUresult,
                   (CUgraphNode, Ptr{CUgraphNode}, Ptr{Csize_t}),
                   hNode, dependentNodes, numDependentNodes)
end

@checked function cuGraphAddDependencies(hGraph, from, to, numDependencies)
    initialize_api()
    ccall((:cuGraphAddDependencies, libcuda()), CUresult,
                   (CUgraph, Ptr{CUgraphNode}, Ptr{CUgraphNode}, Csize_t),
                   hGraph, from, to, numDependencies)
end

@checked function cuGraphRemoveDependencies(hGraph, from, to, numDependencies)
    initialize_api()
    ccall((:cuGraphRemoveDependencies, libcuda()), CUresult,
                   (CUgraph, Ptr{CUgraphNode}, Ptr{CUgraphNode}, Csize_t),
                   hGraph, from, to, numDependencies)
end

@checked function cuGraphDestroyNode(hNode)
    initialize_api()
    ccall((:cuGraphDestroyNode, libcuda()), CUresult,
                   (CUgraphNode,),
                   hNode)
end

@checked function cuGraphInstantiate_v2(phGraphExec, hGraph, phErrorNode, logBuffer,
                                        bufferSize)
    initialize_api()
    ccall((:cuGraphInstantiate_v2, libcuda()), CUresult,
                   (Ptr{CUgraphExec}, CUgraph, Ptr{CUgraphNode}, Cstring, Csize_t),
                   phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize)
end

@checked function cuGraphExecKernelNodeSetParams(hGraphExec, hNode, nodeParams)
    initialize_api()
    ccall((:cuGraphExecKernelNodeSetParams, libcuda()), CUresult,
                   (CUgraphExec, CUgraphNode, Ptr{CUDA_KERNEL_NODE_PARAMS}),
                   hGraphExec, hNode, nodeParams)
end

@checked function cuGraphExecMemcpyNodeSetParams(hGraphExec, hNode, copyParams, ctx)
    initialize_api()
    ccall((:cuGraphExecMemcpyNodeSetParams, libcuda()), CUresult,
                   (CUgraphExec, CUgraphNode, Ptr{CUDA_MEMCPY3D}, CUcontext),
                   hGraphExec, hNode, copyParams, ctx)
end

@checked function cuGraphExecMemsetNodeSetParams(hGraphExec, hNode, memsetParams, ctx)
    initialize_api()
    ccall((:cuGraphExecMemsetNodeSetParams, libcuda()), CUresult,
                   (CUgraphExec, CUgraphNode, Ptr{CUDA_MEMSET_NODE_PARAMS}, CUcontext),
                   hGraphExec, hNode, memsetParams, ctx)
end

@checked function cuGraphExecHostNodeSetParams(hGraphExec, hNode, nodeParams)
    initialize_api()
    ccall((:cuGraphExecHostNodeSetParams, libcuda()), CUresult,
                   (CUgraphExec, CUgraphNode, Ptr{CUDA_HOST_NODE_PARAMS}),
                   hGraphExec, hNode, nodeParams)
end

@checked function cuGraphLaunch(hGraphExec, hStream)
    initialize_api()
    ccall((:cuGraphLaunch_ptsz, libcuda()), CUresult,
                   (CUgraphExec, CUstream),
                   hGraphExec, hStream)
end

@checked function cuGraphExecDestroy(hGraphExec)
    initialize_api()
    ccall((:cuGraphExecDestroy, libcuda()), CUresult,
                   (CUgraphExec,),
                   hGraphExec)
end

@checked function cuGraphDestroy(hGraph)
    initialize_api()
    ccall((:cuGraphDestroy, libcuda()), CUresult,
                   (CUgraph,),
                   hGraph)
end

@checked function cuGraphExecUpdate(hGraphExec, hGraph, hErrorNode_out, updateResult_out)
    initialize_api()
    ccall((:cuGraphExecUpdate, libcuda()), CUresult,
                   (CUgraphExec, CUgraph, Ptr{CUgraphNode}, Ptr{CUgraphExecUpdateResult}),
                   hGraphExec, hGraph, hErrorNode_out, updateResult_out)
end

@checked function cuGraphKernelNodeCopyAttributes(dst, src)
    initialize_api()
    ccall((:cuGraphKernelNodeCopyAttributes, libcuda()), CUresult,
                   (CUgraphNode, CUgraphNode),
                   dst, src)
end

@checked function cuGraphKernelNodeGetAttribute(hNode, attr, value_out)
    initialize_api()
    ccall((:cuGraphKernelNodeGetAttribute, libcuda()), CUresult,
                   (CUgraphNode, CUkernelNodeAttrID, Ptr{CUkernelNodeAttrValue}),
                   hNode, attr, value_out)
end

@checked function cuGraphKernelNodeSetAttribute(hNode, attr, value)
    initialize_api()
    ccall((:cuGraphKernelNodeSetAttribute, libcuda()), CUresult,
                   (CUgraphNode, CUkernelNodeAttrID, Ptr{CUkernelNodeAttrValue}),
                   hNode, attr, value)
end

@checked function cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize,
                                                              dynamicSMemSize)
    initialize_api()
    ccall((:cuOccupancyMaxActiveBlocksPerMultiprocessor, libcuda()), CUresult,
                   (Ptr{Cint}, CUfunction, Cint, Csize_t),
                   numBlocks, func, blockSize, dynamicSMemSize)
end

@checked function cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func,
                                                                       blockSize,
                                                                       dynamicSMemSize,
                                                                       flags)
    initialize_api()
    ccall((:cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, libcuda()), CUresult,
                   (Ptr{Cint}, CUfunction, Cint, Csize_t, UInt32),
                   numBlocks, func, blockSize, dynamicSMemSize, flags)
end

@checked function cuOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func,
                                                   blockSizeToDynamicSMemSize,
                                                   dynamicSMemSize, blockSizeLimit)
    initialize_api()
    ccall((:cuOccupancyMaxPotentialBlockSize, libcuda()), CUresult,
                   (Ptr{Cint}, Ptr{Cint}, CUfunction, CUoccupancyB2DSize, Csize_t, Cint),
                   minGridSize, blockSize, func, blockSizeToDynamicSMemSize,
                   dynamicSMemSize, blockSizeLimit)
end

@checked function cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize, blockSize, func,
                                                            blockSizeToDynamicSMemSize,
                                                            dynamicSMemSize,
                                                            blockSizeLimit, flags)
    initialize_api()
    ccall((:cuOccupancyMaxPotentialBlockSizeWithFlags, libcuda()), CUresult,
                   (Ptr{Cint}, Ptr{Cint}, CUfunction, CUoccupancyB2DSize, Csize_t, Cint,
                    UInt32),
                   minGridSize, blockSize, func, blockSizeToDynamicSMemSize,
                   dynamicSMemSize, blockSizeLimit, flags)
end

@checked function cuOccupancyAvailableDynamicSMemPerBlock(dynamicSmemSize, func, numBlocks,
                                                          blockSize)
    initialize_api()
    ccall((:cuOccupancyAvailableDynamicSMemPerBlock, libcuda()), CUresult,
                   (Ptr{Csize_t}, CUfunction, Cint, Cint),
                   dynamicSmemSize, func, numBlocks, blockSize)
end

@checked function cuTexRefSetArray(hTexRef, hArray, Flags)
    initialize_api()
    ccall((:cuTexRefSetArray, libcuda()), CUresult,
                   (CUtexref, CUarray, UInt32),
                   hTexRef, hArray, Flags)
end

@checked function cuTexRefSetMipmappedArray(hTexRef, hMipmappedArray, Flags)
    initialize_api()
    ccall((:cuTexRefSetMipmappedArray, libcuda()), CUresult,
                   (CUtexref, CUmipmappedArray, UInt32),
                   hTexRef, hMipmappedArray, Flags)
end

@checked function cuTexRefSetAddress_v2(ByteOffset, hTexRef, dptr, bytes)
    initialize_api()
    ccall((:cuTexRefSetAddress_v2, libcuda()), CUresult,
                   (Ptr{Csize_t}, CUtexref, CUdeviceptr, Csize_t),
                   ByteOffset, hTexRef, dptr, bytes)
end

@checked function cuTexRefSetAddress2D_v3(hTexRef, desc, dptr, Pitch)
    initialize_api()
    ccall((:cuTexRefSetAddress2D_v3, libcuda()), CUresult,
                   (CUtexref, Ptr{CUDA_ARRAY_DESCRIPTOR}, CUdeviceptr, Csize_t),
                   hTexRef, desc, dptr, Pitch)
end

@checked function cuTexRefSetFormat(hTexRef, fmt, NumPackedComponents)
    initialize_api()
    ccall((:cuTexRefSetFormat, libcuda()), CUresult,
                   (CUtexref, CUarray_format, Cint),
                   hTexRef, fmt, NumPackedComponents)
end

@checked function cuTexRefSetAddressMode(hTexRef, dim, am)
    initialize_api()
    ccall((:cuTexRefSetAddressMode, libcuda()), CUresult,
                   (CUtexref, Cint, CUaddress_mode),
                   hTexRef, dim, am)
end

@checked function cuTexRefSetFilterMode(hTexRef, fm)
    initialize_api()
    ccall((:cuTexRefSetFilterMode, libcuda()), CUresult,
                   (CUtexref, CUfilter_mode),
                   hTexRef, fm)
end

@checked function cuTexRefSetMipmapFilterMode(hTexRef, fm)
    initialize_api()
    ccall((:cuTexRefSetMipmapFilterMode, libcuda()), CUresult,
                   (CUtexref, CUfilter_mode),
                   hTexRef, fm)
end

@checked function cuTexRefSetMipmapLevelBias(hTexRef, bias)
    initialize_api()
    ccall((:cuTexRefSetMipmapLevelBias, libcuda()), CUresult,
                   (CUtexref, Cfloat),
                   hTexRef, bias)
end

@checked function cuTexRefSetMipmapLevelClamp(hTexRef, minMipmapLevelClamp,
                                              maxMipmapLevelClamp)
    initialize_api()
    ccall((:cuTexRefSetMipmapLevelClamp, libcuda()), CUresult,
                   (CUtexref, Cfloat, Cfloat),
                   hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp)
end

@checked function cuTexRefSetMaxAnisotropy(hTexRef, maxAniso)
    initialize_api()
    ccall((:cuTexRefSetMaxAnisotropy, libcuda()), CUresult,
                   (CUtexref, UInt32),
                   hTexRef, maxAniso)
end

@checked function cuTexRefSetBorderColor(hTexRef, pBorderColor)
    initialize_api()
    ccall((:cuTexRefSetBorderColor, libcuda()), CUresult,
                   (CUtexref, Ptr{Cfloat}),
                   hTexRef, pBorderColor)
end

@checked function cuTexRefSetFlags(hTexRef, Flags)
    initialize_api()
    ccall((:cuTexRefSetFlags, libcuda()), CUresult,
                   (CUtexref, UInt32),
                   hTexRef, Flags)
end

@checked function cuTexRefGetAddress_v2(pdptr, hTexRef)
    initialize_api()
    ccall((:cuTexRefGetAddress_v2, libcuda()), CUresult,
                   (Ptr{CUdeviceptr}, CUtexref),
                   pdptr, hTexRef)
end

@checked function cuTexRefGetArray(phArray, hTexRef)
    initialize_api()
    ccall((:cuTexRefGetArray, libcuda()), CUresult,
                   (Ptr{CUarray}, CUtexref),
                   phArray, hTexRef)
end

@checked function cuTexRefGetMipmappedArray(phMipmappedArray, hTexRef)
    initialize_api()
    ccall((:cuTexRefGetMipmappedArray, libcuda()), CUresult,
                   (Ptr{CUmipmappedArray}, CUtexref),
                   phMipmappedArray, hTexRef)
end

@checked function cuTexRefGetAddressMode(pam, hTexRef, dim)
    initialize_api()
    ccall((:cuTexRefGetAddressMode, libcuda()), CUresult,
                   (Ptr{CUaddress_mode}, CUtexref, Cint),
                   pam, hTexRef, dim)
end

@checked function cuTexRefGetFilterMode(pfm, hTexRef)
    initialize_api()
    ccall((:cuTexRefGetFilterMode, libcuda()), CUresult,
                   (Ptr{CUfilter_mode}, CUtexref),
                   pfm, hTexRef)
end

@checked function cuTexRefGetFormat(pFormat, pNumChannels, hTexRef)
    initialize_api()
    ccall((:cuTexRefGetFormat, libcuda()), CUresult,
                   (Ptr{CUarray_format}, Ptr{Cint}, CUtexref),
                   pFormat, pNumChannels, hTexRef)
end

@checked function cuTexRefGetMipmapFilterMode(pfm, hTexRef)
    initialize_api()
    ccall((:cuTexRefGetMipmapFilterMode, libcuda()), CUresult,
                   (Ptr{CUfilter_mode}, CUtexref),
                   pfm, hTexRef)
end

@checked function cuTexRefGetMipmapLevelBias(pbias, hTexRef)
    initialize_api()
    ccall((:cuTexRefGetMipmapLevelBias, libcuda()), CUresult,
                   (Ptr{Cfloat}, CUtexref),
                   pbias, hTexRef)
end

@checked function cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp, pmaxMipmapLevelClamp,
                                              hTexRef)
    initialize_api()
    ccall((:cuTexRefGetMipmapLevelClamp, libcuda()), CUresult,
                   (Ptr{Cfloat}, Ptr{Cfloat}, CUtexref),
                   pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef)
end

@checked function cuTexRefGetMaxAnisotropy(pmaxAniso, hTexRef)
    initialize_api()
    ccall((:cuTexRefGetMaxAnisotropy, libcuda()), CUresult,
                   (Ptr{Cint}, CUtexref),
                   pmaxAniso, hTexRef)
end

@checked function cuTexRefGetBorderColor(pBorderColor, hTexRef)
    initialize_api()
    ccall((:cuTexRefGetBorderColor, libcuda()), CUresult,
                   (Ptr{Cfloat}, CUtexref),
                   pBorderColor, hTexRef)
end

@checked function cuTexRefGetFlags(pFlags, hTexRef)
    initialize_api()
    ccall((:cuTexRefGetFlags, libcuda()), CUresult,
                   (Ptr{UInt32}, CUtexref),
                   pFlags, hTexRef)
end

@checked function cuTexRefCreate(pTexRef)
    initialize_api()
    ccall((:cuTexRefCreate, libcuda()), CUresult,
                   (Ptr{CUtexref},),
                   pTexRef)
end

@checked function cuTexRefDestroy(hTexRef)
    initialize_api()
    ccall((:cuTexRefDestroy, libcuda()), CUresult,
                   (CUtexref,),
                   hTexRef)
end

@checked function cuSurfRefSetArray(hSurfRef, hArray, Flags)
    initialize_api()
    ccall((:cuSurfRefSetArray, libcuda()), CUresult,
                   (CUsurfref, CUarray, UInt32),
                   hSurfRef, hArray, Flags)
end

@checked function cuSurfRefGetArray(phArray, hSurfRef)
    initialize_api()
    ccall((:cuSurfRefGetArray, libcuda()), CUresult,
                   (Ptr{CUarray}, CUsurfref),
                   phArray, hSurfRef)
end

@checked function cuTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc)
    initialize_api()
    ccall((:cuTexObjectCreate, libcuda()), CUresult,
                   (Ptr{CUtexObject}, Ptr{CUDA_RESOURCE_DESC}, Ptr{CUDA_TEXTURE_DESC},
                    Ptr{CUDA_RESOURCE_VIEW_DESC}),
                   pTexObject, pResDesc, pTexDesc, pResViewDesc)
end

@checked function cuTexObjectDestroy(texObject)
    initialize_api()
    ccall((:cuTexObjectDestroy, libcuda()), CUresult,
                   (CUtexObject,),
                   texObject)
end

@checked function cuTexObjectGetResourceDesc(pResDesc, texObject)
    initialize_api()
    ccall((:cuTexObjectGetResourceDesc, libcuda()), CUresult,
                   (Ptr{CUDA_RESOURCE_DESC}, CUtexObject),
                   pResDesc, texObject)
end

@checked function cuTexObjectGetTextureDesc(pTexDesc, texObject)
    initialize_api()
    ccall((:cuTexObjectGetTextureDesc, libcuda()), CUresult,
                   (Ptr{CUDA_TEXTURE_DESC}, CUtexObject),
                   pTexDesc, texObject)
end

@checked function cuTexObjectGetResourceViewDesc(pResViewDesc, texObject)
    initialize_api()
    ccall((:cuTexObjectGetResourceViewDesc, libcuda()), CUresult,
                   (Ptr{CUDA_RESOURCE_VIEW_DESC}, CUtexObject),
                   pResViewDesc, texObject)
end

@checked function cuSurfObjectCreate(pSurfObject, pResDesc)
    initialize_api()
    ccall((:cuSurfObjectCreate, libcuda()), CUresult,
                   (Ptr{CUsurfObject}, Ptr{CUDA_RESOURCE_DESC}),
                   pSurfObject, pResDesc)
end

@checked function cuSurfObjectDestroy(surfObject)
    initialize_api()
    ccall((:cuSurfObjectDestroy, libcuda()), CUresult,
                   (CUsurfObject,),
                   surfObject)
end

@checked function cuSurfObjectGetResourceDesc(pResDesc, surfObject)
    initialize_api()
    ccall((:cuSurfObjectGetResourceDesc, libcuda()), CUresult,
                   (Ptr{CUDA_RESOURCE_DESC}, CUsurfObject),
                   pResDesc, surfObject)
end

@checked function cuDeviceCanAccessPeer(canAccessPeer, dev, peerDev)
    initialize_api()
    ccall((:cuDeviceCanAccessPeer, libcuda()), CUresult,
                   (Ptr{Cint}, CUdevice, CUdevice),
                   canAccessPeer, dev, peerDev)
end

@checked function cuCtxEnablePeerAccess(peerContext, Flags)
    initialize_api()
    ccall((:cuCtxEnablePeerAccess, libcuda()), CUresult,
                   (CUcontext, UInt32),
                   peerContext, Flags)
end

@checked function cuCtxDisablePeerAccess(peerContext)
    initialize_api()
    ccall((:cuCtxDisablePeerAccess, libcuda()), CUresult,
                   (CUcontext,),
                   peerContext)
end

@checked function cuDeviceGetP2PAttribute(value, attrib, srcDevice, dstDevice)
    initialize_api()
    ccall((:cuDeviceGetP2PAttribute, libcuda()), CUresult,
                   (Ptr{Cint}, CUdevice_P2PAttribute, CUdevice, CUdevice),
                   value, attrib, srcDevice, dstDevice)
end

@checked function cuGraphicsUnregisterResource(resource)
    initialize_api()
    ccall((:cuGraphicsUnregisterResource, libcuda()), CUresult,
                   (CUgraphicsResource,),
                   resource)
end

@checked function cuGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel)
    initialize_api()
    ccall((:cuGraphicsSubResourceGetMappedArray, libcuda()), CUresult,
                   (Ptr{CUarray}, CUgraphicsResource, UInt32, UInt32),
                   pArray, resource, arrayIndex, mipLevel)
end

@checked function cuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray, resource)
    initialize_api()
    ccall((:cuGraphicsResourceGetMappedMipmappedArray, libcuda()), CUresult,
                   (Ptr{CUmipmappedArray}, CUgraphicsResource),
                   pMipmappedArray, resource)
end

@checked function cuGraphicsResourceGetMappedPointer_v2(pDevPtr, pSize, resource)
    initialize_api()
    ccall((:cuGraphicsResourceGetMappedPointer_v2, libcuda()), CUresult,
                   (Ptr{CUdeviceptr}, Ptr{Csize_t}, CUgraphicsResource),
                   pDevPtr, pSize, resource)
end

@checked function cuGraphicsResourceSetMapFlags_v2(resource, flags)
    initialize_api()
    ccall((:cuGraphicsResourceSetMapFlags_v2, libcuda()), CUresult,
                   (CUgraphicsResource, UInt32),
                   resource, flags)
end

@checked function cuGraphicsMapResources(count, resources, hStream)
    initialize_api()
    ccall((:cuGraphicsMapResources_ptsz, libcuda()), CUresult,
                   (UInt32, Ptr{CUgraphicsResource}, CUstream),
                   count, resources, hStream)
end

@checked function cuGraphicsUnmapResources(count, resources, hStream)
    initialize_api()
    ccall((:cuGraphicsUnmapResources_ptsz, libcuda()), CUresult,
                   (UInt32, Ptr{CUgraphicsResource}, CUstream),
                   count, resources, hStream)
end

@checked function cuGetExportTable(ppExportTable, pExportTableId)
    initialize_api()
    ccall((:cuGetExportTable, libcuda()), CUresult,
                   (Ptr{Ptr{Cvoid}}, Ptr{CUuuid}),
                   ppExportTable, pExportTableId)
end

@checked function cuFuncGetModule(hmod, hfunc)
    initialize_api()
    ccall((:cuFuncGetModule, libcuda()), CUresult,
                   (Ptr{CUmodule}, CUfunction),
                   hmod, hfunc)
end
# Julia wrapper for header: cudaProfiler.h
# Automatically generated using Clang.jl

@checked function cuProfilerInitialize(configFile, outputFile, outputMode)
    initialize_api()
    ccall((:cuProfilerInitialize, libcuda()), CUresult,
                   (Cstring, Cstring, CUoutput_mode),
                   configFile, outputFile, outputMode)
end

@checked function cuProfilerStart()
    initialize_api()
    ccall((:cuProfilerStart, libcuda()), CUresult, ())
end

@checked function cuProfilerStop()
    initialize_api()
    ccall((:cuProfilerStop, libcuda()), CUresult, ())
end
