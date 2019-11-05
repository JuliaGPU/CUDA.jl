# Julia wrapper for header: cuda.h
# Automatically generated using Clang.jl


function cuGetErrorString(error, pStr)
    @check ccall((:cuGetErrorString, libcuda), CUresult,
                 (CUresult, Ptr{Cstring}),
                 error, pStr)
end

function cuGetErrorName(error, pStr)
    @check ccall((:cuGetErrorName, libcuda), CUresult,
                 (CUresult, Ptr{Cstring}),
                 error, pStr)
end

function cuInit(Flags)
    @check ccall((:cuInit, libcuda), CUresult,
                 (UInt32,),
                 Flags)
end

function cuDriverGetVersion(driverVersion)
    @check ccall((:cuDriverGetVersion, libcuda), CUresult,
                 (Ptr{Cint},),
                 driverVersion)
end

function cuDeviceGet(device, ordinal)
    @check ccall((:cuDeviceGet, libcuda), CUresult,
                 (Ptr{CUdevice}, Cint),
                 device, ordinal)
end

function cuDeviceGetCount(count)
    @check ccall((:cuDeviceGetCount, libcuda), CUresult,
                 (Ptr{Cint},),
                 count)
end

function cuDeviceGetName(name, len, dev)
    @check ccall((:cuDeviceGetName, libcuda), CUresult,
                 (Cstring, Cint, CUdevice),
                 name, len, dev)
end

function cuDeviceGetUuid(uuid, dev)
    @check ccall((:cuDeviceGetUuid, libcuda), CUresult,
                 (Ptr{CUuuid}, CUdevice),
                 uuid, dev)
end

function cuDeviceTotalMem_v2(bytes, dev)
    @check ccall((:cuDeviceTotalMem_v2, libcuda), CUresult,
                 (Ptr{Csize_t}, CUdevice),
                 bytes, dev)
end

function cuDeviceGetAttribute(pi, attrib, dev)
    @check ccall((:cuDeviceGetAttribute, libcuda), CUresult,
                 (Ptr{Cint}, CUdevice_attribute, CUdevice),
                 pi, attrib, dev)
end

function cuDeviceGetProperties(prop, dev)
    @check ccall((:cuDeviceGetProperties, libcuda), CUresult,
                 (Ptr{CUdevprop}, CUdevice),
                 prop, dev)
end

function cuDeviceComputeCapability(major, minor, dev)
    @check ccall((:cuDeviceComputeCapability, libcuda), CUresult,
                 (Ptr{Cint}, Ptr{Cint}, CUdevice),
                 major, minor, dev)
end

function cuDevicePrimaryCtxRetain(pctx, dev)
    @check ccall((:cuDevicePrimaryCtxRetain, libcuda), CUresult,
                 (Ptr{CUcontext}, CUdevice),
                 pctx, dev)
end

function cuDevicePrimaryCtxRelease(dev)
    @check ccall((:cuDevicePrimaryCtxRelease, libcuda), CUresult,
                 (CUdevice,),
                 dev)
end

function cuDevicePrimaryCtxSetFlags(dev, flags)
    @check ccall((:cuDevicePrimaryCtxSetFlags, libcuda), CUresult,
                 (CUdevice, UInt32),
                 dev, flags)
end

function cuDevicePrimaryCtxGetState(dev, flags, active)
    @check ccall((:cuDevicePrimaryCtxGetState, libcuda), CUresult,
                 (CUdevice, Ptr{UInt32}, Ptr{Cint}),
                 dev, flags, active)
end

function cuDevicePrimaryCtxReset(dev)
    @check ccall((:cuDevicePrimaryCtxReset, libcuda), CUresult,
                 (CUdevice,),
                 dev)
end

function cuCtxCreate_v2(pctx, flags, dev)
    @check ccall((:cuCtxCreate_v2, libcuda), CUresult,
                 (Ptr{CUcontext}, UInt32, CUdevice),
                 pctx, flags, dev)
end

function cuCtxDestroy_v2(ctx)
    @check ccall((:cuCtxDestroy_v2, libcuda), CUresult,
                 (CUcontext,),
                 ctx)
end

function cuCtxPushCurrent_v2(ctx)
    @check ccall((:cuCtxPushCurrent_v2, libcuda), CUresult,
                 (CUcontext,),
                 ctx)
end

function cuCtxPopCurrent_v2(pctx)
    @check ccall((:cuCtxPopCurrent_v2, libcuda), CUresult,
                 (Ptr{CUcontext},),
                 pctx)
end

function cuCtxSetCurrent(ctx)
    @check ccall((:cuCtxSetCurrent, libcuda), CUresult,
                 (CUcontext,),
                 ctx)
end

function cuCtxGetCurrent(pctx)
    @check ccall((:cuCtxGetCurrent, libcuda), CUresult,
                 (Ptr{CUcontext},),
                 pctx)
end

function cuCtxGetDevice(device)
    @check ccall((:cuCtxGetDevice, libcuda), CUresult,
                 (Ptr{CUdevice},),
                 device)
end

function cuCtxGetFlags(flags)
    @check ccall((:cuCtxGetFlags, libcuda), CUresult,
                 (Ptr{UInt32},),
                 flags)
end

function cuCtxSynchronize()
    @check ccall((:cuCtxSynchronize, libcuda), CUresult, ())
end

function cuCtxSetLimit(limit, value)
    @check ccall((:cuCtxSetLimit, libcuda), CUresult,
                 (CUlimit, Csize_t),
                 limit, value)
end

function cuCtxGetLimit(pvalue, limit)
    @check ccall((:cuCtxGetLimit, libcuda), CUresult,
                 (Ptr{Csize_t}, CUlimit),
                 pvalue, limit)
end

function cuCtxGetCacheConfig(pconfig)
    @check ccall((:cuCtxGetCacheConfig, libcuda), CUresult,
                 (Ptr{CUfunc_cache},),
                 pconfig)
end

function cuCtxSetCacheConfig(config)
    @check ccall((:cuCtxSetCacheConfig, libcuda), CUresult,
                 (CUfunc_cache,),
                 config)
end

function cuCtxGetSharedMemConfig(pConfig)
    @check ccall((:cuCtxGetSharedMemConfig, libcuda), CUresult,
                 (Ptr{CUsharedconfig},),
                 pConfig)
end

function cuCtxSetSharedMemConfig(config)
    @check ccall((:cuCtxSetSharedMemConfig, libcuda), CUresult,
                 (CUsharedconfig,),
                 config)
end

function cuCtxGetApiVersion(ctx, version)
    @check ccall((:cuCtxGetApiVersion, libcuda), CUresult,
                 (CUcontext, Ptr{UInt32}),
                 ctx, version)
end

function cuCtxGetStreamPriorityRange(leastPriority, greatestPriority)
    @check ccall((:cuCtxGetStreamPriorityRange, libcuda), CUresult,
                 (Ptr{Cint}, Ptr{Cint}),
                 leastPriority, greatestPriority)
end

function cuCtxAttach(pctx, flags)
    @check ccall((:cuCtxAttach, libcuda), CUresult,
                 (Ptr{CUcontext}, UInt32),
                 pctx, flags)
end

function cuCtxDetach(ctx)
    @check ccall((:cuCtxDetach, libcuda), CUresult,
                 (CUcontext,),
                 ctx)
end

function cuModuleLoad(_module, fname)
    @check ccall((:cuModuleLoad, libcuda), CUresult,
                 (Ptr{CUmodule}, Cstring),
                 _module, fname)
end

function cuModuleLoadData(_module, image)
    @check ccall((:cuModuleLoadData, libcuda), CUresult,
                 (Ptr{CUmodule}, Ptr{Cvoid}),
                 _module, image)
end

function cuModuleLoadDataEx(_module, image, numOptions, options, optionValues)
    @check ccall((:cuModuleLoadDataEx, libcuda), CUresult,
                 (Ptr{CUmodule}, Ptr{Cvoid}, UInt32, Ptr{CUjit_option}, Ptr{Ptr{Cvoid}}),
                 _module, image, numOptions, options, optionValues)
end

function cuModuleLoadFatBinary(_module, fatCubin)
    @check ccall((:cuModuleLoadFatBinary, libcuda), CUresult,
                 (Ptr{CUmodule}, Ptr{Cvoid}),
                 _module, fatCubin)
end

function cuModuleUnload(hmod)
    @check ccall((:cuModuleUnload, libcuda), CUresult,
                 (CUmodule,),
                 hmod)
end

function cuModuleGetFunction(hfunc, hmod, name)
    @check ccall((:cuModuleGetFunction, libcuda), CUresult,
                 (Ptr{CUfunction}, CUmodule, Cstring),
                 hfunc, hmod, name)
end

function cuModuleGetGlobal_v2(dptr, bytes, hmod, name)
    @check ccall((:cuModuleGetGlobal_v2, libcuda), CUresult,
                 (Ptr{CUdeviceptr}, Ptr{Csize_t}, CUmodule, Cstring),
                 dptr, bytes, hmod, name)
end

function cuModuleGetTexRef(pTexRef, hmod, name)
    @check ccall((:cuModuleGetTexRef, libcuda), CUresult,
                 (Ptr{CUtexref}, CUmodule, Cstring),
                 pTexRef, hmod, name)
end

function cuModuleGetSurfRef(pSurfRef, hmod, name)
    @check ccall((:cuModuleGetSurfRef, libcuda), CUresult,
                 (Ptr{CUsurfref}, CUmodule, Cstring),
                 pSurfRef, hmod, name)
end

function cuLinkCreate_v2(numOptions, options, optionValues, stateOut)
    @check ccall((:cuLinkCreate_v2, libcuda), CUresult,
                 (UInt32, Ptr{CUjit_option}, Ptr{Ptr{Cvoid}}, Ptr{CUlinkState}),
                 numOptions, options, optionValues, stateOut)
end

function cuLinkAddData_v2(state, type, data, size, name, numOptions, options, optionValues)
    @check ccall((:cuLinkAddData_v2, libcuda), CUresult,
                 (CUlinkState, CUjitInputType, Ptr{Cvoid}, Csize_t, Cstring, UInt32,
                  Ptr{CUjit_option}, Ptr{Ptr{Cvoid}}),
                 state, type, data, size, name, numOptions, options, optionValues)
end

function cuLinkAddFile_v2(state, type, path, numOptions, options, optionValues)
    @check ccall((:cuLinkAddFile_v2, libcuda), CUresult,
                 (CUlinkState, CUjitInputType, Cstring, UInt32, Ptr{CUjit_option},
                  Ptr{Ptr{Cvoid}}),
                 state, type, path, numOptions, options, optionValues)
end

function cuLinkComplete(state, cubinOut, sizeOut)
    @check ccall((:cuLinkComplete, libcuda), CUresult,
                 (CUlinkState, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}),
                 state, cubinOut, sizeOut)
end

function cuLinkDestroy(state)
    @check ccall((:cuLinkDestroy, libcuda), CUresult,
                 (CUlinkState,),
                 state)
end

function cuMemGetInfo_v2(free, total)
    @check ccall((:cuMemGetInfo_v2, libcuda), CUresult,
                 (Ptr{Csize_t}, Ptr{Csize_t}),
                 free, total)
end

function cuMemAlloc_v2(dptr, bytesize)
    @check ccall((:cuMemAlloc_v2, libcuda), CUresult,
                 (Ptr{CUdeviceptr}, Csize_t),
                 dptr, bytesize)
end

function cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes)
    @check ccall((:cuMemAllocPitch_v2, libcuda), CUresult,
                 (Ptr{CUdeviceptr}, Ptr{Csize_t}, Csize_t, Csize_t, UInt32),
                 dptr, pPitch, WidthInBytes, Height, ElementSizeBytes)
end

function cuMemFree_v2(dptr)
    @check ccall((:cuMemFree_v2, libcuda), CUresult,
                 (CUdeviceptr,),
                 dptr)
end

function cuMemGetAddressRange_v2(pbase, psize, dptr)
    @check ccall((:cuMemGetAddressRange_v2, libcuda), CUresult,
                 (Ptr{CUdeviceptr}, Ptr{Csize_t}, CUdeviceptr),
                 pbase, psize, dptr)
end

function cuMemAllocHost_v2(pp, bytesize)
    @check ccall((:cuMemAllocHost_v2, libcuda), CUresult,
                 (Ptr{Ptr{Cvoid}}, Csize_t),
                 pp, bytesize)
end

function cuMemFreeHost(p)
    @check ccall((:cuMemFreeHost, libcuda), CUresult,
                 (Ptr{Cvoid},),
                 p)
end

function cuMemHostAlloc(pp, bytesize, Flags)
    @check ccall((:cuMemHostAlloc, libcuda), CUresult,
                 (Ptr{Ptr{Cvoid}}, Csize_t, UInt32),
                 pp, bytesize, Flags)
end

function cuMemHostGetDevicePointer_v2(pdptr, p, Flags)
    @check ccall((:cuMemHostGetDevicePointer_v2, libcuda), CUresult,
                 (Ptr{CUdeviceptr}, Ptr{Cvoid}, UInt32),
                 pdptr, p, Flags)
end

function cuMemHostGetFlags(pFlags, p)
    @check ccall((:cuMemHostGetFlags, libcuda), CUresult,
                 (Ptr{UInt32}, Ptr{Cvoid}),
                 pFlags, p)
end

function cuMemAllocManaged(dptr, bytesize, flags)
    @check ccall((:cuMemAllocManaged, libcuda), CUresult,
                 (Ptr{CUdeviceptr}, Csize_t, UInt32),
                 dptr, bytesize, flags)
end

function cuDeviceGetByPCIBusId(dev, pciBusId)
    @check ccall((:cuDeviceGetByPCIBusId, libcuda), CUresult,
                 (Ptr{CUdevice}, Cstring),
                 dev, pciBusId)
end

function cuDeviceGetPCIBusId(pciBusId, len, dev)
    @check ccall((:cuDeviceGetPCIBusId, libcuda), CUresult,
                 (Cstring, Cint, CUdevice),
                 pciBusId, len, dev)
end

function cuIpcGetEventHandle(pHandle, event)
    @check ccall((:cuIpcGetEventHandle, libcuda), CUresult,
                 (Ptr{CUipcEventHandle}, CUevent),
                 pHandle, event)
end

function cuIpcOpenEventHandle(phEvent, handle)
    @check ccall((:cuIpcOpenEventHandle, libcuda), CUresult,
                 (Ptr{CUevent}, CUipcEventHandle),
                 phEvent, handle)
end

function cuIpcGetMemHandle(pHandle, dptr)
    @check ccall((:cuIpcGetMemHandle, libcuda), CUresult,
                 (Ptr{CUipcMemHandle}, CUdeviceptr),
                 pHandle, dptr)
end

function cuIpcOpenMemHandle(pdptr, handle, Flags)
    @check ccall((:cuIpcOpenMemHandle, libcuda), CUresult,
                 (Ptr{CUdeviceptr}, CUipcMemHandle, UInt32),
                 pdptr, handle, Flags)
end

function cuIpcCloseMemHandle(dptr)
    @check ccall((:cuIpcCloseMemHandle, libcuda), CUresult,
                 (CUdeviceptr,),
                 dptr)
end

function cuMemHostRegister_v2(p, bytesize, Flags)
    @check ccall((:cuMemHostRegister_v2, libcuda), CUresult,
                 (Ptr{Cvoid}, Csize_t, UInt32),
                 p, bytesize, Flags)
end

function cuMemHostUnregister(p)
    @check ccall((:cuMemHostUnregister, libcuda), CUresult,
                 (Ptr{Cvoid},),
                 p)
end

function cuMemcpy(dst, src, ByteCount)
    @check ccall((:cuMemcpy, libcuda), CUresult,
                 (CUdeviceptr, CUdeviceptr, Csize_t),
                 dst, src, ByteCount)
end

function cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount)
    @check ccall((:cuMemcpyPeer, libcuda), CUresult,
                 (CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, Csize_t),
                 dstDevice, dstContext, srcDevice, srcContext, ByteCount)
end

function cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount)
    @check ccall((:cuMemcpyHtoD_v2, libcuda), CUresult,
                 (CUdeviceptr, Ptr{Cvoid}, Csize_t),
                 dstDevice, srcHost, ByteCount)
end

function cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount)
    @check ccall((:cuMemcpyDtoH_v2, libcuda), CUresult,
                 (Ptr{Cvoid}, CUdeviceptr, Csize_t),
                 dstHost, srcDevice, ByteCount)
end

function cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount)
    @check ccall((:cuMemcpyDtoD_v2, libcuda), CUresult,
                 (CUdeviceptr, CUdeviceptr, Csize_t),
                 dstDevice, srcDevice, ByteCount)
end

function cuMemcpyDtoA_v2(dstArray, dstOffset, srcDevice, ByteCount)
    @check ccall((:cuMemcpyDtoA_v2, libcuda), CUresult,
                 (CUarray, Csize_t, CUdeviceptr, Csize_t),
                 dstArray, dstOffset, srcDevice, ByteCount)
end

function cuMemcpyAtoD_v2(dstDevice, srcArray, srcOffset, ByteCount)
    @check ccall((:cuMemcpyAtoD_v2, libcuda), CUresult,
                 (CUdeviceptr, CUarray, Csize_t, Csize_t),
                 dstDevice, srcArray, srcOffset, ByteCount)
end

function cuMemcpyHtoA_v2(dstArray, dstOffset, srcHost, ByteCount)
    @check ccall((:cuMemcpyHtoA_v2, libcuda), CUresult,
                 (CUarray, Csize_t, Ptr{Cvoid}, Csize_t),
                 dstArray, dstOffset, srcHost, ByteCount)
end

function cuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount)
    @check ccall((:cuMemcpyAtoH_v2, libcuda), CUresult,
                 (Ptr{Cvoid}, CUarray, Csize_t, Csize_t),
                 dstHost, srcArray, srcOffset, ByteCount)
end

function cuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount)
    @check ccall((:cuMemcpyAtoA_v2, libcuda), CUresult,
                 (CUarray, Csize_t, CUarray, Csize_t, Csize_t),
                 dstArray, dstOffset, srcArray, srcOffset, ByteCount)
end

function cuMemcpy2D_v2(pCopy)
    @check ccall((:cuMemcpy2D_v2, libcuda), CUresult,
                 (Ptr{CUDA_MEMCPY2D},),
                 pCopy)
end

function cuMemcpy2DUnaligned_v2(pCopy)
    @check ccall((:cuMemcpy2DUnaligned_v2, libcuda), CUresult,
                 (Ptr{CUDA_MEMCPY2D},),
                 pCopy)
end

function cuMemcpy3D_v2(pCopy)
    @check ccall((:cuMemcpy3D_v2, libcuda), CUresult,
                 (Ptr{CUDA_MEMCPY3D},),
                 pCopy)
end

function cuMemcpy3DPeer(pCopy)
    @check ccall((:cuMemcpy3DPeer, libcuda), CUresult,
                 (Ptr{CUDA_MEMCPY3D_PEER},),
                 pCopy)
end

function cuMemcpyAsync(dst, src, ByteCount, hStream)
    @check ccall((:cuMemcpyAsync, libcuda), CUresult,
                 (CUdeviceptr, CUdeviceptr, Csize_t, CUstream),
                 dst, src, ByteCount, hStream)
end

function cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream)
    @check ccall((:cuMemcpyPeerAsync, libcuda), CUresult,
                 (CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, Csize_t, CUstream),
                 dstDevice, dstContext, srcDevice, srcContext, ByteCount, hStream)
end

function cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream)
    @check ccall((:cuMemcpyHtoDAsync_v2, libcuda), CUresult,
                 (CUdeviceptr, Ptr{Cvoid}, Csize_t, CUstream),
                 dstDevice, srcHost, ByteCount, hStream)
end

function cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream)
    @check ccall((:cuMemcpyDtoHAsync_v2, libcuda), CUresult,
                 (Ptr{Cvoid}, CUdeviceptr, Csize_t, CUstream),
                 dstHost, srcDevice, ByteCount, hStream)
end

function cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream)
    @check ccall((:cuMemcpyDtoDAsync_v2, libcuda), CUresult,
                 (CUdeviceptr, CUdeviceptr, Csize_t, CUstream),
                 dstDevice, srcDevice, ByteCount, hStream)
end

function cuMemcpyHtoAAsync_v2(dstArray, dstOffset, srcHost, ByteCount, hStream)
    @check ccall((:cuMemcpyHtoAAsync_v2, libcuda), CUresult,
                 (CUarray, Csize_t, Ptr{Cvoid}, Csize_t, CUstream),
                 dstArray, dstOffset, srcHost, ByteCount, hStream)
end

function cuMemcpyAtoHAsync_v2(dstHost, srcArray, srcOffset, ByteCount, hStream)
    @check ccall((:cuMemcpyAtoHAsync_v2, libcuda), CUresult,
                 (Ptr{Cvoid}, CUarray, Csize_t, Csize_t, CUstream),
                 dstHost, srcArray, srcOffset, ByteCount, hStream)
end

function cuMemcpy2DAsync_v2(pCopy, hStream)
    @check ccall((:cuMemcpy2DAsync_v2, libcuda), CUresult,
                 (Ptr{CUDA_MEMCPY2D}, CUstream),
                 pCopy, hStream)
end

function cuMemcpy3DAsync_v2(pCopy, hStream)
    @check ccall((:cuMemcpy3DAsync_v2, libcuda), CUresult,
                 (Ptr{CUDA_MEMCPY3D}, CUstream),
                 pCopy, hStream)
end

function cuMemcpy3DPeerAsync(pCopy, hStream)
    @check ccall((:cuMemcpy3DPeerAsync, libcuda), CUresult,
                 (Ptr{CUDA_MEMCPY3D_PEER}, CUstream),
                 pCopy, hStream)
end

function cuMemsetD8_v2(dstDevice, uc, N)
    @check ccall((:cuMemsetD8_v2, libcuda), CUresult,
                 (CUdeviceptr, Cuchar, Csize_t),
                 dstDevice, uc, N)
end

function cuMemsetD16_v2(dstDevice, us, N)
    @check ccall((:cuMemsetD16_v2, libcuda), CUresult,
                 (CUdeviceptr, UInt16, Csize_t),
                 dstDevice, us, N)
end

function cuMemsetD32_v2(dstDevice, ui, N)
    @check ccall((:cuMemsetD32_v2, libcuda), CUresult,
                 (CUdeviceptr, UInt32, Csize_t),
                 dstDevice, ui, N)
end

function cuMemsetD2D8_v2(dstDevice, dstPitch, uc, Width, Height)
    @check ccall((:cuMemsetD2D8_v2, libcuda), CUresult,
                 (CUdeviceptr, Csize_t, Cuchar, Csize_t, Csize_t),
                 dstDevice, dstPitch, uc, Width, Height)
end

function cuMemsetD2D16_v2(dstDevice, dstPitch, us, Width, Height)
    @check ccall((:cuMemsetD2D16_v2, libcuda), CUresult,
                 (CUdeviceptr, Csize_t, UInt16, Csize_t, Csize_t),
                 dstDevice, dstPitch, us, Width, Height)
end

function cuMemsetD2D32_v2(dstDevice, dstPitch, ui, Width, Height)
    @check ccall((:cuMemsetD2D32_v2, libcuda), CUresult,
                 (CUdeviceptr, Csize_t, UInt32, Csize_t, Csize_t),
                 dstDevice, dstPitch, ui, Width, Height)
end

function cuMemsetD8Async(dstDevice, uc, N, hStream)
    @check ccall((:cuMemsetD8Async, libcuda), CUresult,
                 (CUdeviceptr, Cuchar, Csize_t, CUstream),
                 dstDevice, uc, N, hStream)
end

function cuMemsetD16Async(dstDevice, us, N, hStream)
    @check ccall((:cuMemsetD16Async, libcuda), CUresult,
                 (CUdeviceptr, UInt16, Csize_t, CUstream),
                 dstDevice, us, N, hStream)
end

function cuMemsetD32Async(dstDevice, ui, N, hStream)
    @check ccall((:cuMemsetD32Async, libcuda), CUresult,
                 (CUdeviceptr, UInt32, Csize_t, CUstream),
                 dstDevice, ui, N, hStream)
end

function cuMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream)
    @check ccall((:cuMemsetD2D8Async, libcuda), CUresult,
                 (CUdeviceptr, Csize_t, Cuchar, Csize_t, Csize_t, CUstream),
                 dstDevice, dstPitch, uc, Width, Height, hStream)
end

function cuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream)
    @check ccall((:cuMemsetD2D16Async, libcuda), CUresult,
                 (CUdeviceptr, Csize_t, UInt16, Csize_t, Csize_t, CUstream),
                 dstDevice, dstPitch, us, Width, Height, hStream)
end

function cuMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream)
    @check ccall((:cuMemsetD2D32Async, libcuda), CUresult,
                 (CUdeviceptr, Csize_t, UInt32, Csize_t, Csize_t, CUstream),
                 dstDevice, dstPitch, ui, Width, Height, hStream)
end

function cuArrayCreate_v2(pHandle, pAllocateArray)
    @check ccall((:cuArrayCreate_v2, libcuda), CUresult,
                 (Ptr{CUarray}, Ptr{CUDA_ARRAY_DESCRIPTOR}),
                 pHandle, pAllocateArray)
end

function cuArrayGetDescriptor_v2(pArrayDescriptor, hArray)
    @check ccall((:cuArrayGetDescriptor_v2, libcuda), CUresult,
                 (Ptr{CUDA_ARRAY_DESCRIPTOR}, CUarray),
                 pArrayDescriptor, hArray)
end

function cuArrayDestroy(hArray)
    @check ccall((:cuArrayDestroy, libcuda), CUresult,
                 (CUarray,),
                 hArray)
end

function cuArray3DCreate_v2(pHandle, pAllocateArray)
    @check ccall((:cuArray3DCreate_v2, libcuda), CUresult,
                 (Ptr{CUarray}, Ptr{CUDA_ARRAY3D_DESCRIPTOR}),
                 pHandle, pAllocateArray)
end

function cuArray3DGetDescriptor_v2(pArrayDescriptor, hArray)
    @check ccall((:cuArray3DGetDescriptor_v2, libcuda), CUresult,
                 (Ptr{CUDA_ARRAY3D_DESCRIPTOR}, CUarray),
                 pArrayDescriptor, hArray)
end

function cuMipmappedArrayCreate(pHandle, pMipmappedArrayDesc, numMipmapLevels)
    @check ccall((:cuMipmappedArrayCreate, libcuda), CUresult,
                 (Ptr{CUmipmappedArray}, Ptr{CUDA_ARRAY3D_DESCRIPTOR}, UInt32),
                 pHandle, pMipmappedArrayDesc, numMipmapLevels)
end

function cuMipmappedArrayGetLevel(pLevelArray, hMipmappedArray, level)
    @check ccall((:cuMipmappedArrayGetLevel, libcuda), CUresult,
                 (Ptr{CUarray}, CUmipmappedArray, UInt32),
                 pLevelArray, hMipmappedArray, level)
end

function cuMipmappedArrayDestroy(hMipmappedArray)
    @check ccall((:cuMipmappedArrayDestroy, libcuda), CUresult,
                 (CUmipmappedArray,),
                 hMipmappedArray)
end

function cuPointerGetAttribute(data, attribute, ptr)
    @check ccall((:cuPointerGetAttribute, libcuda), CUresult,
                 (Ptr{Cvoid}, CUpointer_attribute, CUdeviceptr),
                 data, attribute, ptr)
end

function cuMemPrefetchAsync(devPtr, count, dstDevice, hStream)
    @check ccall((:cuMemPrefetchAsync, libcuda), CUresult,
                 (CUdeviceptr, Csize_t, CUdevice, CUstream),
                 devPtr, count, dstDevice, hStream)
end

function cuMemAdvise(devPtr, count, advice, device)
    @check ccall((:cuMemAdvise, libcuda), CUresult,
                 (CUdeviceptr, Csize_t, CUmem_advise, CUdevice),
                 devPtr, count, advice, device)
end

function cuMemRangeGetAttribute(data, dataSize, attribute, devPtr, count)
    @check ccall((:cuMemRangeGetAttribute, libcuda), CUresult,
                 (Ptr{Cvoid}, Csize_t, CUmem_range_attribute, CUdeviceptr, Csize_t),
                 data, dataSize, attribute, devPtr, count)
end

function cuMemRangeGetAttributes(data, dataSizes, attributes, numAttributes, devPtr, count)
    @check ccall((:cuMemRangeGetAttributes, libcuda), CUresult,
                 (Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Ptr{CUmem_range_attribute}, Csize_t,
                  CUdeviceptr, Csize_t),
                 data, dataSizes, attributes, numAttributes, devPtr, count)
end

function cuPointerSetAttribute(value, attribute, ptr)
    @check ccall((:cuPointerSetAttribute, libcuda), CUresult,
                 (Ptr{Cvoid}, CUpointer_attribute, CUdeviceptr),
                 value, attribute, ptr)
end

function cuPointerGetAttributes(numAttributes, attributes, data, ptr)
    @check ccall((:cuPointerGetAttributes, libcuda), CUresult,
                 (UInt32, Ptr{CUpointer_attribute}, Ptr{Ptr{Cvoid}}, CUdeviceptr),
                 numAttributes, attributes, data, ptr)
end

function cuStreamCreate(phStream, Flags)
    @check ccall((:cuStreamCreate, libcuda), CUresult,
                 (Ptr{CUstream}, UInt32),
                 phStream, Flags)
end

function cuStreamCreateWithPriority(phStream, flags, priority)
    @check ccall((:cuStreamCreateWithPriority, libcuda), CUresult,
                 (Ptr{CUstream}, UInt32, Cint),
                 phStream, flags, priority)
end

function cuStreamGetPriority(hStream, priority)
    @check ccall((:cuStreamGetPriority, libcuda), CUresult,
                 (CUstream, Ptr{Cint}),
                 hStream, priority)
end

function cuStreamGetFlags(hStream, flags)
    @check ccall((:cuStreamGetFlags, libcuda), CUresult,
                 (CUstream, Ptr{UInt32}),
                 hStream, flags)
end

function cuStreamGetCtx(hStream, pctx)
    @check ccall((:cuStreamGetCtx, libcuda), CUresult,
                 (CUstream, Ptr{CUcontext}),
                 hStream, pctx)
end

function cuStreamWaitEvent(hStream, hEvent, Flags)
    @check ccall((:cuStreamWaitEvent, libcuda), CUresult,
                 (CUstream, CUevent, UInt32),
                 hStream, hEvent, Flags)
end

function cuStreamAddCallback(hStream, callback, userData, flags)
    @check ccall((:cuStreamAddCallback, libcuda), CUresult,
                 (CUstream, CUstreamCallback, Ptr{Cvoid}, UInt32),
                 hStream, callback, userData, flags)
end

function cuStreamBeginCapture_v2(hStream, mode)
    @check ccall((:cuStreamBeginCapture_v2, libcuda), CUresult,
                 (CUstream, CUstreamCaptureMode),
                 hStream, mode)
end

function cuThreadExchangeStreamCaptureMode(mode)
    @check ccall((:cuThreadExchangeStreamCaptureMode, libcuda), CUresult,
                 (Ptr{CUstreamCaptureMode},),
                 mode)
end

function cuStreamEndCapture(hStream, phGraph)
    @check ccall((:cuStreamEndCapture, libcuda), CUresult,
                 (CUstream, Ptr{CUgraph}),
                 hStream, phGraph)
end

function cuStreamIsCapturing(hStream, captureStatus)
    @check ccall((:cuStreamIsCapturing, libcuda), CUresult,
                 (CUstream, Ptr{CUstreamCaptureStatus}),
                 hStream, captureStatus)
end

function cuStreamGetCaptureInfo(hStream, captureStatus, id)
    @check ccall((:cuStreamGetCaptureInfo, libcuda), CUresult,
                 (CUstream, Ptr{CUstreamCaptureStatus}, Ptr{cuuint64_t}),
                 hStream, captureStatus, id)
end

function cuStreamAttachMemAsync(hStream, dptr, length, flags)
    @check ccall((:cuStreamAttachMemAsync, libcuda), CUresult,
                 (CUstream, CUdeviceptr, Csize_t, UInt32),
                 hStream, dptr, length, flags)
end

function cuStreamQuery(hStream)
    @check ccall((:cuStreamQuery, libcuda), CUresult,
                 (CUstream,),
                 hStream)
end

function cuStreamSynchronize(hStream)
    @check ccall((:cuStreamSynchronize, libcuda), CUresult,
                 (CUstream,),
                 hStream)
end

function cuStreamDestroy_v2(hStream)
    @check ccall((:cuStreamDestroy_v2, libcuda), CUresult,
                 (CUstream,),
                 hStream)
end

function cuEventCreate(phEvent, Flags)
    @check ccall((:cuEventCreate, libcuda), CUresult,
                 (Ptr{CUevent}, UInt32),
                 phEvent, Flags)
end

function cuEventRecord(hEvent, hStream)
    @check ccall((:cuEventRecord, libcuda), CUresult,
                 (CUevent, CUstream),
                 hEvent, hStream)
end

function cuEventQuery(hEvent)
    @check ccall((:cuEventQuery, libcuda), CUresult,
                 (CUevent,),
                 hEvent)
end

function cuEventSynchronize(hEvent)
    @check ccall((:cuEventSynchronize, libcuda), CUresult,
                 (CUevent,),
                 hEvent)
end

function cuEventDestroy_v2(hEvent)
    @check ccall((:cuEventDestroy_v2, libcuda), CUresult,
                 (CUevent,),
                 hEvent)
end

function cuEventElapsedTime(pMilliseconds, hStart, hEnd)
    @check ccall((:cuEventElapsedTime, libcuda), CUresult,
                 (Ptr{Cfloat}, CUevent, CUevent),
                 pMilliseconds, hStart, hEnd)
end

function cuImportExternalMemory(extMem_out, memHandleDesc)
    @check ccall((:cuImportExternalMemory, libcuda), CUresult,
                 (Ptr{CUexternalMemory}, Ptr{CUDA_EXTERNAL_MEMORY_HANDLE_DESC}),
                 extMem_out, memHandleDesc)
end

function cuExternalMemoryGetMappedBuffer(devPtr, extMem, bufferDesc)
    @check ccall((:cuExternalMemoryGetMappedBuffer, libcuda), CUresult,
                 (Ptr{CUdeviceptr}, CUexternalMemory,
                  Ptr{CUDA_EXTERNAL_MEMORY_BUFFER_DESC}),
                 devPtr, extMem, bufferDesc)
end

function cuExternalMemoryGetMappedMipmappedArray(mipmap, extMem, mipmapDesc)
    @check ccall((:cuExternalMemoryGetMappedMipmappedArray, libcuda), CUresult,
                 (Ptr{CUmipmappedArray}, CUexternalMemory,
                  Ptr{CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC}),
                 mipmap, extMem, mipmapDesc)
end

function cuDestroyExternalMemory(extMem)
    @check ccall((:cuDestroyExternalMemory, libcuda), CUresult,
                 (CUexternalMemory,),
                 extMem)
end

function cuImportExternalSemaphore(extSem_out, semHandleDesc)
    @check ccall((:cuImportExternalSemaphore, libcuda), CUresult,
                 (Ptr{CUexternalSemaphore}, Ptr{CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC}),
                 extSem_out, semHandleDesc)
end

function cuSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)
    @check ccall((:cuSignalExternalSemaphoresAsync, libcuda), CUresult,
                 (Ptr{CUexternalSemaphore}, Ptr{CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS},
                  UInt32, CUstream),
                 extSemArray, paramsArray, numExtSems, stream)
end

function cuWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems, stream)
    @check ccall((:cuWaitExternalSemaphoresAsync, libcuda), CUresult,
                 (Ptr{CUexternalSemaphore}, Ptr{CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS},
                  UInt32, CUstream),
                 extSemArray, paramsArray, numExtSems, stream)
end

function cuDestroyExternalSemaphore(extSem)
    @check ccall((:cuDestroyExternalSemaphore, libcuda), CUresult,
                 (CUexternalSemaphore,),
                 extSem)
end

function cuStreamWaitValue32(stream, addr, value, flags)
    @check ccall((:cuStreamWaitValue32, libcuda), CUresult,
                 (CUstream, CUdeviceptr, cuuint32_t, UInt32),
                 stream, addr, value, flags)
end

function cuStreamWaitValue64(stream, addr, value, flags)
    @check ccall((:cuStreamWaitValue64, libcuda), CUresult,
                 (CUstream, CUdeviceptr, cuuint64_t, UInt32),
                 stream, addr, value, flags)
end

function cuStreamWriteValue32(stream, addr, value, flags)
    @check ccall((:cuStreamWriteValue32, libcuda), CUresult,
                 (CUstream, CUdeviceptr, cuuint32_t, UInt32),
                 stream, addr, value, flags)
end

function cuStreamWriteValue64(stream, addr, value, flags)
    @check ccall((:cuStreamWriteValue64, libcuda), CUresult,
                 (CUstream, CUdeviceptr, cuuint64_t, UInt32),
                 stream, addr, value, flags)
end

function cuFuncGetAttribute(pi, attrib, hfunc)
    @check ccall((:cuFuncGetAttribute, libcuda), CUresult,
                 (Ptr{Cint}, CUfunction_attribute, CUfunction),
                 pi, attrib, hfunc)
end

function cuFuncSetAttribute(hfunc, attrib, value)
    @check ccall((:cuFuncSetAttribute, libcuda), CUresult,
                 (CUfunction, CUfunction_attribute, Cint),
                 hfunc, attrib, value)
end

function cuFuncSetCacheConfig(hfunc, config)
    @check ccall((:cuFuncSetCacheConfig, libcuda), CUresult,
                 (CUfunction, CUfunc_cache),
                 hfunc, config)
end

function cuFuncSetSharedMemConfig(hfunc, config)
    @check ccall((:cuFuncSetSharedMemConfig, libcuda), CUresult,
                 (CUfunction, CUsharedconfig),
                 hfunc, config)
end

function cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                        sharedMemBytes, hStream, kernelParams, extra)
    @check ccall((:cuLaunchKernel, libcuda), CUresult,
                 (CUfunction, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                  CUstream, Ptr{Ptr{Cvoid}}, Ptr{Ptr{Cvoid}}),
                 f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                 sharedMemBytes, hStream, kernelParams, extra)
end

function cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                                   blockDimZ, sharedMemBytes, hStream, kernelParams)
    @check ccall((:cuLaunchCooperativeKernel, libcuda), CUresult,
                 (CUfunction, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                  CUstream, Ptr{Ptr{Cvoid}}),
                 f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                 sharedMemBytes, hStream, kernelParams)
end

function cuLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags)
    @check ccall((:cuLaunchCooperativeKernelMultiDevice, libcuda), CUresult,
                 (Ptr{CUDA_LAUNCH_PARAMS}, UInt32, UInt32),
                 launchParamsList, numDevices, flags)
end

function cuLaunchHostFunc(hStream, fn, userData)
    @check ccall((:cuLaunchHostFunc, libcuda), CUresult,
                 (CUstream, CUhostFn, Ptr{Cvoid}),
                 hStream, fn, userData)
end

function cuFuncSetBlockShape(hfunc, x, y, z)
    @check ccall((:cuFuncSetBlockShape, libcuda), CUresult,
                 (CUfunction, Cint, Cint, Cint),
                 hfunc, x, y, z)
end

function cuFuncSetSharedSize(hfunc, bytes)
    @check ccall((:cuFuncSetSharedSize, libcuda), CUresult,
                 (CUfunction, UInt32),
                 hfunc, bytes)
end

function cuParamSetSize(hfunc, numbytes)
    @check ccall((:cuParamSetSize, libcuda), CUresult,
                 (CUfunction, UInt32),
                 hfunc, numbytes)
end

function cuParamSeti(hfunc, offset, value)
    @check ccall((:cuParamSeti, libcuda), CUresult,
                 (CUfunction, Cint, UInt32),
                 hfunc, offset, value)
end

function cuParamSetf(hfunc, offset, value)
    @check ccall((:cuParamSetf, libcuda), CUresult,
                 (CUfunction, Cint, Cfloat),
                 hfunc, offset, value)
end

function cuParamSetv(hfunc, offset, ptr, numbytes)
    @check ccall((:cuParamSetv, libcuda), CUresult,
                 (CUfunction, Cint, Ptr{Cvoid}, UInt32),
                 hfunc, offset, ptr, numbytes)
end

function cuLaunch(f)
    @check ccall((:cuLaunch, libcuda), CUresult,
                 (CUfunction,),
                 f)
end

function cuLaunchGrid(f, grid_width, grid_height)
    @check ccall((:cuLaunchGrid, libcuda), CUresult,
                 (CUfunction, Cint, Cint),
                 f, grid_width, grid_height)
end

function cuLaunchGridAsync(f, grid_width, grid_height, hStream)
    @check ccall((:cuLaunchGridAsync, libcuda), CUresult,
                 (CUfunction, Cint, Cint, CUstream),
                 f, grid_width, grid_height, hStream)
end

function cuParamSetTexRef(hfunc, texunit, hTexRef)
    @check ccall((:cuParamSetTexRef, libcuda), CUresult,
                 (CUfunction, Cint, CUtexref),
                 hfunc, texunit, hTexRef)
end

function cuGraphCreate(phGraph, flags)
    @check ccall((:cuGraphCreate, libcuda), CUresult,
                 (Ptr{CUgraph}, UInt32),
                 phGraph, flags)
end

function cuGraphAddKernelNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    @check ccall((:cuGraphAddKernelNode, libcuda), CUresult,
                 (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t,
                  Ptr{CUDA_KERNEL_NODE_PARAMS}),
                 phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
end

function cuGraphKernelNodeGetParams(hNode, nodeParams)
    @check ccall((:cuGraphKernelNodeGetParams, libcuda), CUresult,
                 (CUgraphNode, Ptr{CUDA_KERNEL_NODE_PARAMS}),
                 hNode, nodeParams)
end

function cuGraphKernelNodeSetParams(hNode, nodeParams)
    @check ccall((:cuGraphKernelNodeSetParams, libcuda), CUresult,
                 (CUgraphNode, Ptr{CUDA_KERNEL_NODE_PARAMS}),
                 hNode, nodeParams)
end

function cuGraphAddMemcpyNode(phGraphNode, hGraph, dependencies, numDependencies,
                              copyParams, ctx)
    @check ccall((:cuGraphAddMemcpyNode, libcuda), CUresult,
                 (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t,
                  Ptr{CUDA_MEMCPY3D}, CUcontext),
                 phGraphNode, hGraph, dependencies, numDependencies, copyParams, ctx)
end

function cuGraphMemcpyNodeGetParams(hNode, nodeParams)
    @check ccall((:cuGraphMemcpyNodeGetParams, libcuda), CUresult,
                 (CUgraphNode, Ptr{CUDA_MEMCPY3D}),
                 hNode, nodeParams)
end

function cuGraphMemcpyNodeSetParams(hNode, nodeParams)
    @check ccall((:cuGraphMemcpyNodeSetParams, libcuda), CUresult,
                 (CUgraphNode, Ptr{CUDA_MEMCPY3D}),
                 hNode, nodeParams)
end

function cuGraphAddMemsetNode(phGraphNode, hGraph, dependencies, numDependencies,
                              memsetParams, ctx)
    @check ccall((:cuGraphAddMemsetNode, libcuda), CUresult,
                 (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t,
                  Ptr{CUDA_MEMSET_NODE_PARAMS}, CUcontext),
                 phGraphNode, hGraph, dependencies, numDependencies, memsetParams, ctx)
end

function cuGraphMemsetNodeGetParams(hNode, nodeParams)
    @check ccall((:cuGraphMemsetNodeGetParams, libcuda), CUresult,
                 (CUgraphNode, Ptr{CUDA_MEMSET_NODE_PARAMS}),
                 hNode, nodeParams)
end

function cuGraphMemsetNodeSetParams(hNode, nodeParams)
    @check ccall((:cuGraphMemsetNodeSetParams, libcuda), CUresult,
                 (CUgraphNode, Ptr{CUDA_MEMSET_NODE_PARAMS}),
                 hNode, nodeParams)
end

function cuGraphAddHostNode(phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
    @check ccall((:cuGraphAddHostNode, libcuda), CUresult,
                 (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t,
                  Ptr{CUDA_HOST_NODE_PARAMS}),
                 phGraphNode, hGraph, dependencies, numDependencies, nodeParams)
end

function cuGraphHostNodeGetParams(hNode, nodeParams)
    @check ccall((:cuGraphHostNodeGetParams, libcuda), CUresult,
                 (CUgraphNode, Ptr{CUDA_HOST_NODE_PARAMS}),
                 hNode, nodeParams)
end

function cuGraphHostNodeSetParams(hNode, nodeParams)
    @check ccall((:cuGraphHostNodeSetParams, libcuda), CUresult,
                 (CUgraphNode, Ptr{CUDA_HOST_NODE_PARAMS}),
                 hNode, nodeParams)
end

function cuGraphAddChildGraphNode(phGraphNode, hGraph, dependencies, numDependencies,
                                  childGraph)
    @check ccall((:cuGraphAddChildGraphNode, libcuda), CUresult,
                 (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t, CUgraph),
                 phGraphNode, hGraph, dependencies, numDependencies, childGraph)
end

function cuGraphChildGraphNodeGetGraph(hNode, phGraph)
    @check ccall((:cuGraphChildGraphNodeGetGraph, libcuda), CUresult,
                 (CUgraphNode, Ptr{CUgraph}),
                 hNode, phGraph)
end

function cuGraphAddEmptyNode(phGraphNode, hGraph, dependencies, numDependencies)
    @check ccall((:cuGraphAddEmptyNode, libcuda), CUresult,
                 (Ptr{CUgraphNode}, CUgraph, Ptr{CUgraphNode}, Csize_t),
                 phGraphNode, hGraph, dependencies, numDependencies)
end

function cuGraphClone(phGraphClone, originalGraph)
    @check ccall((:cuGraphClone, libcuda), CUresult,
                 (Ptr{CUgraph}, CUgraph),
                 phGraphClone, originalGraph)
end

function cuGraphNodeFindInClone(phNode, hOriginalNode, hClonedGraph)
    @check ccall((:cuGraphNodeFindInClone, libcuda), CUresult,
                 (Ptr{CUgraphNode}, CUgraphNode, CUgraph),
                 phNode, hOriginalNode, hClonedGraph)
end

function cuGraphNodeGetType(hNode, type)
    @check ccall((:cuGraphNodeGetType, libcuda), CUresult,
                 (CUgraphNode, Ptr{CUgraphNodeType}),
                 hNode, type)
end

function cuGraphGetNodes(hGraph, nodes, numNodes)
    @check ccall((:cuGraphGetNodes, libcuda), CUresult,
                 (CUgraph, Ptr{CUgraphNode}, Ptr{Csize_t}),
                 hGraph, nodes, numNodes)
end

function cuGraphGetRootNodes(hGraph, rootNodes, numRootNodes)
    @check ccall((:cuGraphGetRootNodes, libcuda), CUresult,
                 (CUgraph, Ptr{CUgraphNode}, Ptr{Csize_t}),
                 hGraph, rootNodes, numRootNodes)
end

function cuGraphGetEdges(hGraph, from, to, numEdges)
    @check ccall((:cuGraphGetEdges, libcuda), CUresult,
                 (CUgraph, Ptr{CUgraphNode}, Ptr{CUgraphNode}, Ptr{Csize_t}),
                 hGraph, from, to, numEdges)
end

function cuGraphNodeGetDependencies(hNode, dependencies, numDependencies)
    @check ccall((:cuGraphNodeGetDependencies, libcuda), CUresult,
                 (CUgraphNode, Ptr{CUgraphNode}, Ptr{Csize_t}),
                 hNode, dependencies, numDependencies)
end

function cuGraphNodeGetDependentNodes(hNode, dependentNodes, numDependentNodes)
    @check ccall((:cuGraphNodeGetDependentNodes, libcuda), CUresult,
                 (CUgraphNode, Ptr{CUgraphNode}, Ptr{Csize_t}),
                 hNode, dependentNodes, numDependentNodes)
end

function cuGraphAddDependencies(hGraph, from, to, numDependencies)
    @check ccall((:cuGraphAddDependencies, libcuda), CUresult,
                 (CUgraph, Ptr{CUgraphNode}, Ptr{CUgraphNode}, Csize_t),
                 hGraph, from, to, numDependencies)
end

function cuGraphRemoveDependencies(hGraph, from, to, numDependencies)
    @check ccall((:cuGraphRemoveDependencies, libcuda), CUresult,
                 (CUgraph, Ptr{CUgraphNode}, Ptr{CUgraphNode}, Csize_t),
                 hGraph, from, to, numDependencies)
end

function cuGraphDestroyNode(hNode)
    @check ccall((:cuGraphDestroyNode, libcuda), CUresult,
                 (CUgraphNode,),
                 hNode)
end

function cuGraphInstantiate(phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize)
    @check ccall((:cuGraphInstantiate, libcuda), CUresult,
                 (Ptr{CUgraphExec}, CUgraph, Ptr{CUgraphNode}, Cstring, Csize_t),
                 phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize)
end

function cuGraphExecKernelNodeSetParams(hGraphExec, hNode, nodeParams)
    @check ccall((:cuGraphExecKernelNodeSetParams, libcuda), CUresult,
                 (CUgraphExec, CUgraphNode, Ptr{CUDA_KERNEL_NODE_PARAMS}),
                 hGraphExec, hNode, nodeParams)
end

function cuGraphLaunch(hGraphExec, hStream)
    @check ccall((:cuGraphLaunch, libcuda), CUresult,
                 (CUgraphExec, CUstream),
                 hGraphExec, hStream)
end

function cuGraphExecDestroy(hGraphExec)
    @check ccall((:cuGraphExecDestroy, libcuda), CUresult,
                 (CUgraphExec,),
                 hGraphExec)
end

function cuGraphDestroy(hGraph)
    @check ccall((:cuGraphDestroy, libcuda), CUresult,
                 (CUgraph,),
                 hGraph)
end

function cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize,
                                                     dynamicSMemSize)
    @check ccall((:cuOccupancyMaxActiveBlocksPerMultiprocessor, libcuda), CUresult,
                 (Ptr{Cint}, CUfunction, Cint, Csize_t),
                 numBlocks, func, blockSize, dynamicSMemSize)
end

function cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize,
                                                              dynamicSMemSize, flags)
    @check ccall((:cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, libcuda), CUresult,
                 (Ptr{Cint}, CUfunction, Cint, Csize_t, UInt32),
                 numBlocks, func, blockSize, dynamicSMemSize, flags)
end

function cuOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func,
                                          blockSizeToDynamicSMemSize, dynamicSMemSize,
                                          blockSizeLimit)
    @check ccall((:cuOccupancyMaxPotentialBlockSize, libcuda), CUresult,
                 (Ptr{Cint}, Ptr{Cint}, CUfunction, CUoccupancyB2DSize, Csize_t, Cint),
                 minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize,
                 blockSizeLimit)
end

function cuOccupancyMaxPotentialBlockSizeWithFlags(minGridSize, blockSize, func,
                                                   blockSizeToDynamicSMemSize,
                                                   dynamicSMemSize, blockSizeLimit, flags)
    @check ccall((:cuOccupancyMaxPotentialBlockSizeWithFlags, libcuda), CUresult,
                 (Ptr{Cint}, Ptr{Cint}, CUfunction, CUoccupancyB2DSize, Csize_t, Cint,
                  UInt32),
                 minGridSize, blockSize, func, blockSizeToDynamicSMemSize, dynamicSMemSize,
                 blockSizeLimit, flags)
end

function cuTexRefSetArray(hTexRef, hArray, Flags)
    @check ccall((:cuTexRefSetArray, libcuda), CUresult,
                 (CUtexref, CUarray, UInt32),
                 hTexRef, hArray, Flags)
end

function cuTexRefSetMipmappedArray(hTexRef, hMipmappedArray, Flags)
    @check ccall((:cuTexRefSetMipmappedArray, libcuda), CUresult,
                 (CUtexref, CUmipmappedArray, UInt32),
                 hTexRef, hMipmappedArray, Flags)
end

function cuTexRefSetAddress_v2(ByteOffset, hTexRef, dptr, bytes)
    @check ccall((:cuTexRefSetAddress_v2, libcuda), CUresult,
                 (Ptr{Csize_t}, CUtexref, CUdeviceptr, Csize_t),
                 ByteOffset, hTexRef, dptr, bytes)
end

function cuTexRefSetAddress2D_v3(hTexRef, desc, dptr, Pitch)
    @check ccall((:cuTexRefSetAddress2D_v3, libcuda), CUresult,
                 (CUtexref, Ptr{CUDA_ARRAY_DESCRIPTOR}, CUdeviceptr, Csize_t),
                 hTexRef, desc, dptr, Pitch)
end

function cuTexRefSetFormat(hTexRef, fmt, NumPackedComponents)
    @check ccall((:cuTexRefSetFormat, libcuda), CUresult,
                 (CUtexref, CUarray_format, Cint),
                 hTexRef, fmt, NumPackedComponents)
end

function cuTexRefSetAddressMode(hTexRef, dim, am)
    @check ccall((:cuTexRefSetAddressMode, libcuda), CUresult,
                 (CUtexref, Cint, CUaddress_mode),
                 hTexRef, dim, am)
end

function cuTexRefSetFilterMode(hTexRef, fm)
    @check ccall((:cuTexRefSetFilterMode, libcuda), CUresult,
                 (CUtexref, CUfilter_mode),
                 hTexRef, fm)
end

function cuTexRefSetMipmapFilterMode(hTexRef, fm)
    @check ccall((:cuTexRefSetMipmapFilterMode, libcuda), CUresult,
                 (CUtexref, CUfilter_mode),
                 hTexRef, fm)
end

function cuTexRefSetMipmapLevelBias(hTexRef, bias)
    @check ccall((:cuTexRefSetMipmapLevelBias, libcuda), CUresult,
                 (CUtexref, Cfloat),
                 hTexRef, bias)
end

function cuTexRefSetMipmapLevelClamp(hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp)
    @check ccall((:cuTexRefSetMipmapLevelClamp, libcuda), CUresult,
                 (CUtexref, Cfloat, Cfloat),
                 hTexRef, minMipmapLevelClamp, maxMipmapLevelClamp)
end

function cuTexRefSetMaxAnisotropy(hTexRef, maxAniso)
    @check ccall((:cuTexRefSetMaxAnisotropy, libcuda), CUresult,
                 (CUtexref, UInt32),
                 hTexRef, maxAniso)
end

function cuTexRefSetBorderColor(hTexRef, pBorderColor)
    @check ccall((:cuTexRefSetBorderColor, libcuda), CUresult,
                 (CUtexref, Ptr{Cfloat}),
                 hTexRef, pBorderColor)
end

function cuTexRefSetFlags(hTexRef, Flags)
    @check ccall((:cuTexRefSetFlags, libcuda), CUresult,
                 (CUtexref, UInt32),
                 hTexRef, Flags)
end

function cuTexRefGetAddress_v2(pdptr, hTexRef)
    @check ccall((:cuTexRefGetAddress_v2, libcuda), CUresult,
                 (Ptr{CUdeviceptr}, CUtexref),
                 pdptr, hTexRef)
end

function cuTexRefGetArray(phArray, hTexRef)
    @check ccall((:cuTexRefGetArray, libcuda), CUresult,
                 (Ptr{CUarray}, CUtexref),
                 phArray, hTexRef)
end

function cuTexRefGetMipmappedArray(phMipmappedArray, hTexRef)
    @check ccall((:cuTexRefGetMipmappedArray, libcuda), CUresult,
                 (Ptr{CUmipmappedArray}, CUtexref),
                 phMipmappedArray, hTexRef)
end

function cuTexRefGetAddressMode(pam, hTexRef, dim)
    @check ccall((:cuTexRefGetAddressMode, libcuda), CUresult,
                 (Ptr{CUaddress_mode}, CUtexref, Cint),
                 pam, hTexRef, dim)
end

function cuTexRefGetFilterMode(pfm, hTexRef)
    @check ccall((:cuTexRefGetFilterMode, libcuda), CUresult,
                 (Ptr{CUfilter_mode}, CUtexref),
                 pfm, hTexRef)
end

function cuTexRefGetFormat(pFormat, pNumChannels, hTexRef)
    @check ccall((:cuTexRefGetFormat, libcuda), CUresult,
                 (Ptr{CUarray_format}, Ptr{Cint}, CUtexref),
                 pFormat, pNumChannels, hTexRef)
end

function cuTexRefGetMipmapFilterMode(pfm, hTexRef)
    @check ccall((:cuTexRefGetMipmapFilterMode, libcuda), CUresult,
                 (Ptr{CUfilter_mode}, CUtexref),
                 pfm, hTexRef)
end

function cuTexRefGetMipmapLevelBias(pbias, hTexRef)
    @check ccall((:cuTexRefGetMipmapLevelBias, libcuda), CUresult,
                 (Ptr{Cfloat}, CUtexref),
                 pbias, hTexRef)
end

function cuTexRefGetMipmapLevelClamp(pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef)
    @check ccall((:cuTexRefGetMipmapLevelClamp, libcuda), CUresult,
                 (Ptr{Cfloat}, Ptr{Cfloat}, CUtexref),
                 pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef)
end

function cuTexRefGetMaxAnisotropy(pmaxAniso, hTexRef)
    @check ccall((:cuTexRefGetMaxAnisotropy, libcuda), CUresult,
                 (Ptr{Cint}, CUtexref),
                 pmaxAniso, hTexRef)
end

function cuTexRefGetBorderColor(pBorderColor, hTexRef)
    @check ccall((:cuTexRefGetBorderColor, libcuda), CUresult,
                 (Ptr{Cfloat}, CUtexref),
                 pBorderColor, hTexRef)
end

function cuTexRefGetFlags(pFlags, hTexRef)
    @check ccall((:cuTexRefGetFlags, libcuda), CUresult,
                 (Ptr{UInt32}, CUtexref),
                 pFlags, hTexRef)
end

function cuTexRefCreate(pTexRef)
    @check ccall((:cuTexRefCreate, libcuda), CUresult,
                 (Ptr{CUtexref},),
                 pTexRef)
end

function cuTexRefDestroy(hTexRef)
    @check ccall((:cuTexRefDestroy, libcuda), CUresult,
                 (CUtexref,),
                 hTexRef)
end

function cuSurfRefSetArray(hSurfRef, hArray, Flags)
    @check ccall((:cuSurfRefSetArray, libcuda), CUresult,
                 (CUsurfref, CUarray, UInt32),
                 hSurfRef, hArray, Flags)
end

function cuSurfRefGetArray(phArray, hSurfRef)
    @check ccall((:cuSurfRefGetArray, libcuda), CUresult,
                 (Ptr{CUarray}, CUsurfref),
                 phArray, hSurfRef)
end

function cuTexObjectCreate(pTexObject, pResDesc, pTexDesc, pResViewDesc)
    @check ccall((:cuTexObjectCreate, libcuda), CUresult,
                 (Ptr{CUtexObject}, Ptr{CUDA_RESOURCE_DESC}, Ptr{CUDA_TEXTURE_DESC},
                  Ptr{CUDA_RESOURCE_VIEW_DESC}),
                 pTexObject, pResDesc, pTexDesc, pResViewDesc)
end

function cuTexObjectDestroy(texObject)
    @check ccall((:cuTexObjectDestroy, libcuda), CUresult,
                 (CUtexObject,),
                 texObject)
end

function cuTexObjectGetResourceDesc(pResDesc, texObject)
    @check ccall((:cuTexObjectGetResourceDesc, libcuda), CUresult,
                 (Ptr{CUDA_RESOURCE_DESC}, CUtexObject),
                 pResDesc, texObject)
end

function cuTexObjectGetTextureDesc(pTexDesc, texObject)
    @check ccall((:cuTexObjectGetTextureDesc, libcuda), CUresult,
                 (Ptr{CUDA_TEXTURE_DESC}, CUtexObject),
                 pTexDesc, texObject)
end

function cuTexObjectGetResourceViewDesc(pResViewDesc, texObject)
    @check ccall((:cuTexObjectGetResourceViewDesc, libcuda), CUresult,
                 (Ptr{CUDA_RESOURCE_VIEW_DESC}, CUtexObject),
                 pResViewDesc, texObject)
end

function cuSurfObjectCreate(pSurfObject, pResDesc)
    @check ccall((:cuSurfObjectCreate, libcuda), CUresult,
                 (Ptr{CUsurfObject}, Ptr{CUDA_RESOURCE_DESC}),
                 pSurfObject, pResDesc)
end

function cuSurfObjectDestroy(surfObject)
    @check ccall((:cuSurfObjectDestroy, libcuda), CUresult,
                 (CUsurfObject,),
                 surfObject)
end

function cuSurfObjectGetResourceDesc(pResDesc, surfObject)
    @check ccall((:cuSurfObjectGetResourceDesc, libcuda), CUresult,
                 (Ptr{CUDA_RESOURCE_DESC}, CUsurfObject),
                 pResDesc, surfObject)
end

function cuDeviceCanAccessPeer(canAccessPeer, dev, peerDev)
    @check ccall((:cuDeviceCanAccessPeer, libcuda), CUresult,
                 (Ptr{Cint}, CUdevice, CUdevice),
                 canAccessPeer, dev, peerDev)
end

function cuCtxEnablePeerAccess(peerContext, Flags)
    @check ccall((:cuCtxEnablePeerAccess, libcuda), CUresult,
                 (CUcontext, UInt32),
                 peerContext, Flags)
end

function cuCtxDisablePeerAccess(peerContext)
    @check ccall((:cuCtxDisablePeerAccess, libcuda), CUresult,
                 (CUcontext,),
                 peerContext)
end

function cuDeviceGetP2PAttribute(value, attrib, srcDevice, dstDevice)
    @check ccall((:cuDeviceGetP2PAttribute, libcuda), CUresult,
                 (Ptr{Cint}, CUdevice_P2PAttribute, CUdevice, CUdevice),
                 value, attrib, srcDevice, dstDevice)
end

function cuGraphicsUnregisterResource(resource)
    @check ccall((:cuGraphicsUnregisterResource, libcuda), CUresult,
                 (CUgraphicsResource,),
                 resource)
end

function cuGraphicsSubResourceGetMappedArray(pArray, resource, arrayIndex, mipLevel)
    @check ccall((:cuGraphicsSubResourceGetMappedArray, libcuda), CUresult,
                 (Ptr{CUarray}, CUgraphicsResource, UInt32, UInt32),
                 pArray, resource, arrayIndex, mipLevel)
end

function cuGraphicsResourceGetMappedMipmappedArray(pMipmappedArray, resource)
    @check ccall((:cuGraphicsResourceGetMappedMipmappedArray, libcuda), CUresult,
                 (Ptr{CUmipmappedArray}, CUgraphicsResource),
                 pMipmappedArray, resource)
end

function cuGraphicsResourceGetMappedPointer_v2(pDevPtr, pSize, resource)
    @check ccall((:cuGraphicsResourceGetMappedPointer_v2, libcuda), CUresult,
                 (Ptr{CUdeviceptr}, Ptr{Csize_t}, CUgraphicsResource),
                 pDevPtr, pSize, resource)
end

function cuGraphicsResourceSetMapFlags_v2(resource, flags)
    @check ccall((:cuGraphicsResourceSetMapFlags_v2, libcuda), CUresult,
                 (CUgraphicsResource, UInt32),
                 resource, flags)
end

function cuGraphicsMapResources(count, resources, hStream)
    @check ccall((:cuGraphicsMapResources, libcuda), CUresult,
                 (UInt32, Ptr{CUgraphicsResource}, CUstream),
                 count, resources, hStream)
end

function cuGraphicsUnmapResources(count, resources, hStream)
    @check ccall((:cuGraphicsUnmapResources, libcuda), CUresult,
                 (UInt32, Ptr{CUgraphicsResource}, CUstream),
                 count, resources, hStream)
end

function cuGetExportTable(ppExportTable, pExportTableId)
    @check ccall((:cuGetExportTable, libcuda), CUresult,
                 (Ptr{Ptr{Cvoid}}, Ptr{CUuuid}),
                 ppExportTable, pExportTableId)
end
# Julia wrapper for header: cudaProfiler.h
# Automatically generated using Clang.jl


function cuProfilerInitialize(configFile, outputFile, outputMode)
    @check ccall((:cuProfilerInitialize, libcuda), CUresult,
                 (Cstring, Cstring, CUoutput_mode),
                 configFile, outputFile, outputMode)
end

function cuProfilerStart()
    @check ccall((:cuProfilerStart, libcuda), CUresult, ())
end

function cuProfilerStop()
    @check ccall((:cuProfilerStop, libcuda), CUresult, ())
end
