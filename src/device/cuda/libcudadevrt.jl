# Julia wrapper for header: cuda_device_runtime_api.h
# Automatically generated using Clang.jl

# Not fully automated, since Clang.jl can't handle the `extern "C"`

# TODO: checking

function cudaDeviceGetAttribute(value, attr, device)
    ccall("extern cudaDeviceGetAttribute", llvmcall, cudaError_t,
          (Ptr{Cint}, cudaDeviceAttr, Cint),
          value, attr, device)
end

function cudaDeviceGetLimit(pValue, limit)
    ccall("extern cudaDeviceGetLimit", llvmcall, cudaError_t,
          (Ptr{Csize_t}, cudaLimit),
          pValue, limit)
end

function cudaDeviceGetCacheConfig(pCacheConfig)
    ccall("extern cudaDeviceGetCacheConfig", llvmcall, cudaError_t,
          (Ptr{cudaFuncCache},),
          pCacheConfig)
end

function cudaDeviceGetSharedMemConfig(pConfig)
    ccall("extern cudaDeviceGetSharedMemConfig", llvmcall, cudaError_t,
          (Ptr{cudaSharedMemConfig},),
          pConfig)
end

function cudaDeviceSynchronize()
    ccall("extern cudaDeviceSynchronize", llvmcall, cudaError_t, ())
end

function cudaGetLastError()
    ccall("extern cudaGetLastError", llvmcall, cudaError_t, ())
end

function cudaPeekAtLastError()
    ccall("extern cudaPeekAtLastError", llvmcall, cudaError_t, ())
end

function cudaGetErrorString(error)
    ccall("extern cudaGetErrorString", llvmcall, Cstring,
          (cudaError_t,),
          error)
end

function cudaGetErrorName(error)
    ccall("extern cudaGetErrorName", llvmcall, Cstring,
          (cudaError_t,),
          error)
end

function cudaGetDeviceCount(count)
    ccall("extern cudaGetDeviceCount", llvmcall, cudaError_t,
          (Ptr{Cint},),
          count)
end

function cudaGetDevice(device)
    ccall("extern cudaGetDevice", llvmcall, cudaError_t,
          (Ptr{Cint},),
          device)
end

function cudaStreamCreateWithFlags(pStream, flags)
    ccall("extern cudaStreamCreateWithFlags", llvmcall, cudaError_t,
          (Ptr{cudaStream_t}, UInt32),
          pStream, flags)
end

function cudaStreamDestroy(stream)
    ccall("extern cudaStreamDestroy", llvmcall, cudaError_t,
          (cudaStream_t,),
          stream)
end

function cudaStreamWaitEvent(stream, event, flags)
    ccall("extern cudaStreamWaitEvent", llvmcall, cudaError_t,
          (cudaStream_t, cudaEvent_t, UInt32),
          stream, event, flags)
end

function cudaStreamWaitEvent_ptsz(stream, event, flags)
    ccall("extern cudaStreamWaitEvent_ptsz", llvmcall, cudaError_t,
          (cudaStream_t, cudaEvent_t, UInt32),
          stream, event, flags)
end

function cudaEventCreateWithFlags(event, flags)
    ccall("extern cudaEventCreateWithFlags", llvmcall, cudaError_t,
          (Ptr{cudaEvent_t}, UInt32),
          event, flags)
end

function cudaEventRecord(event, stream)
    ccall("extern cudaEventRecord", llvmcall, cudaError_t,
          (cudaEvent_t, cudaStream_t),
          event, stream)
end

function cudaEventRecord_ptsz(event, stream)
    ccall("extern cudaEventRecord_ptsz", llvmcall, cudaError_t,
          (cudaEvent_t, cudaStream_t),
          event, stream)
end

function cudaEventDestroy(event)
    ccall("extern cudaEventDestroy", llvmcall, cudaError_t,
          (cudaEvent_t,),
          event)
end

function cudaFuncGetAttributes(attr, func)
    ccall("extern cudaFuncGetAttributes", llvmcall, cudaError_t,
          (Ptr{cudaFuncAttributes}, Ptr{Cvoid}),
          attr, func)
end

function cudaFree(devPtr)
    ccall("extern cudaFree", llvmcall, cudaError_t,
          (Ptr{Cvoid},),
          devPtr)
end

function cudaMalloc(devPtr, size)
    ccall("extern cudaMalloc", llvmcall, cudaError_t,
          (Ptr{Ptr{Cvoid}}, Csize_t),
          devPtr, size)
end

function cudaMemcpyAsync(dst, src, count, kind, stream)
    ccall("extern cudaMemcpyAsync", llvmcall, cudaError_t,
          (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, cudaMemcpyKind, cudaStream_t),
          dst, src, count, kind, stream)
end

function cudaMemcpyAsync_ptsz(dst, src, count, kind, stream)
    ccall("extern cudaMemcpyAsync_ptsz", llvmcall, cudaError_t,
          (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, cudaMemcpyKind, cudaStream_t),
          dst, src, count, kind, stream)
end

function cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream)
    ccall("extern cudaMemcpy2DAsync", llvmcall, cudaError_t,
          (Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, cudaMemcpyKind,
           cudaStream_t),
          dst, dpitch, src, spitch, width, height, kind, stream)
end

function cudaMemcpy2DAsync_ptsz(dst, dpitch, src, spitch, width, height, kind, stream)
    ccall("extern cudaMemcpy2DAsync_ptsz", llvmcall, cudaError_t,
          (Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Csize_t, Csize_t, Csize_t, cudaMemcpyKind,
           cudaStream_t),
          dst, dpitch, src, spitch, width, height, kind, stream)
end

function cudaMemcpy3DAsync(p, stream)
    ccall("extern cudaMemcpy3DAsync", llvmcall, cudaError_t,
          (Ptr{cudaMemcpy3DParms}, cudaStream_t),
          p, stream)
end

function cudaMemcpy3DAsync_ptsz(p, stream)
    ccall("extern cudaMemcpy3DAsync_ptsz", llvmcall, cudaError_t,
          (Ptr{cudaMemcpy3DParms}, cudaStream_t),
          p, stream)
end

function cudaMemsetAsync(devPtr, value, count, stream)
    ccall("extern cudaMemsetAsync", llvmcall, cudaError_t,
          (Ptr{Cvoid}, Cint, Csize_t, cudaStream_t),
          devPtr, value, count, stream)
end

function cudaMemsetAsync_ptsz(devPtr, value, count, stream)
    ccall("extern cudaMemsetAsync_ptsz", llvmcall, cudaError_t,
          (Ptr{Cvoid}, Cint, Csize_t, cudaStream_t),
          devPtr, value, count, stream)
end

function cudaMemset2DAsync(devPtr, pitch, value, width, height, stream)
    ccall("extern cudaMemset2DAsync", llvmcall, cudaError_t,
          (Ptr{Cvoid}, Csize_t, Cint, Csize_t, Csize_t, cudaStream_t),
          devPtr, pitch, value, width, height, stream)
end

function cudaMemset2DAsync_ptsz(devPtr, pitch, value, width, height, stream)
    ccall("extern cudaMemset2DAsync_ptsz", llvmcall, cudaError_t,
          (Ptr{Cvoid}, Csize_t, Cint, Csize_t, Csize_t, cudaStream_t),
          devPtr, pitch, value, width, height, stream)
end

function cudaMemset3DAsync(pitchedDevPtr, value, extent, stream)
    ccall("extern cudaMemset3DAsync", llvmcall, cudaError_t,
          (cudaPitchedPtr, Cint, cudaExtent, cudaStream_t),
          pitchedDevPtr, value, extent, stream)
end

function cudaMemset3DAsync_ptsz(pitchedDevPtr, value, extent, stream)
    ccall("extern cudaMemset3DAsync_ptsz", llvmcall, cudaError_t,
          (cudaPitchedPtr, Cint, cudaExtent, cudaStream_t),
          pitchedDevPtr, value, extent, stream)
end

function cudaRuntimeGetVersion(runtimeVersion)
    ccall("extern cudaRuntimeGetVersion", llvmcall, cudaError_t,
          (Ptr{Cint},),
          runtimeVersion)
end

function cudaGetParameterBuffer(alignment, size)
    ccall("extern cudaGetParameterBuffer", llvmcall, Ptr{Cvoid},
          (Csize_t, Csize_t),
          alignment, size)
end

function cudaGetParameterBufferV2(func, gridDimension, blockDimension, sharedMemSize)
    ccall("extern cudaGetParameterBufferV2", llvmcall, Ptr{Cvoid},
          (Ptr{Cvoid}, dim3, dim3, UInt32),
          func, gridDimension, blockDimension, sharedMemSize)
end

function cudaLaunchDevice_ptsz(func, parameterBuffer, gridDimension, blockDimension,
                               sharedMemSize, stream)
    ccall("extern cudaLaunchDevice_ptsz", llvmcall, cudaError_t,
          (Ptr{Cvoid}, Ptr{Cvoid}, dim3, dim3, UInt32, cudaStream_t),
          func, parameterBuffer, gridDimension, blockDimension, sharedMemSize, stream)
end

function cudaLaunchDeviceV2_ptsz(parameterBuffer, stream)
    ccall("extern cudaLaunchDeviceV2_ptsz", llvmcall, cudaError_t,
          (Ptr{Cvoid}, cudaStream_t),
          parameterBuffer, stream)
end

function cudaLaunchDevice(func, parameterBuffer, gridDimension, blockDimension,
                          sharedMemSize, stream)
    ccall("extern cudaLaunchDevice", llvmcall, cudaError_t,
          (Ptr{Cvoid}, Ptr{Cvoid}, dim3, dim3, UInt32, cudaStream_t),
          func, parameterBuffer, gridDimension, blockDimension, sharedMemSize, stream)
end

function cudaLaunchDeviceV2(parameterBuffer, stream)
    ccall("extern cudaLaunchDeviceV2", llvmcall, cudaError_t,
          (Ptr{Cvoid}, cudaStream_t),
          parameterBuffer, stream)
end

function cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize,
                                                       dynamicSmemSize)
    ccall("extern cudaOccupancyMaxActiveBlocksPerMultiprocessor", llvmcall, cudaError_t,
          (Ptr{Cint}, Ptr{Cvoid}, Cint, Csize_t),
          numBlocks, func, blockSize, dynamicSmemSize)
end

function cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize,
                                                                dynamicSmemSize, flags)
    ccall("extern cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", llvmcall, cudaError_t,
          (Ptr{Cint}, Ptr{Cvoid}, Cint, Csize_t, UInt32),
          numBlocks, func, blockSize, dynamicSmemSize, flags)
end

function cudaCGGetIntrinsicHandle(scope)
    ccall("extern cudaCGGetIntrinsicHandle", llvmcall, Culonglong,
          (cudaCGScope,),
          scope)
end

function cudaCGSynchronize(handle, flags)
    ccall("extern cudaCGSynchronize", llvmcall, cudaError_t,
          (Culonglong, UInt32),
          handle, flags)
end

function cudaCGSynchronizeGrid(handle, flags)
    ccall("extern cudaCGSynchronizeGrid", llvmcall, cudaError_t,
          (Culonglong, UInt32),
          handle, flags)
end

function cudaCGGetSize(numThreads, numGrids, handle)
    ccall("extern cudaCGGetSize", llvmcall, cudaError_t,
          (Ptr{UInt32}, Ptr{UInt32}, Culonglong),
          numThreads, numGrids, handle)
end

function cudaCGGetRank(threadRank, gridRank, handle)
    ccall("extern cudaCGGetRank", llvmcall, cudaError_t,
          (Ptr{UInt32}, Ptr{UInt32}, Culonglong),
          threadRank, gridRank, handle)
end
