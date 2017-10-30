# Julia wrapper for header: /usr/local/cuda/include/cusolver.h

#helper functions
function cusolverDnCreate(handle)
  statuscheck(ccall( (:cusolverDnCreate, libcusolver), cusolverStatus_t, (Ptr{cusolverDnHandle_t},), handle))
end
function cusolverDnDestroy(handle)
  statuscheck(ccall( (:cusolverDnDestroy, libcusolver), cusolverStatus_t, (cusolverDnHandle_t,), handle))
end
function cusolverDnSetStream(handle, streamId)
  statuscheck(ccall( (:cusolverDnSetStream, libcusolver), cusolverStatus_t, (cusolverDnHandle_t, cudaStream_t), handle, streamId))
end
function cusolverDnGetStream(handle, streamId)
  statuscheck(ccall( (:cusolverDnGetStream, libcusolver), cusolverStatus_t, (cusolverDnHandle_t, Ptr{cudaStream_t}), handle, streamId))
end
function cusolverSpCreate(handle)
  statuscheck(ccall( (:cusolverSpCreate, libcusolver), cusolverStatus_t, (Ptr{cusolverSpHandle_t},), handle))
end
function cusolverSpDestroy(handle)
  statuscheck(ccall( (:cusolverSpDestroy, libcusolver), cusolverStatus_t, (cusolverSpHandle_t,), handle))
end
function cusolverSpSetStream(handle, streamId)
  statuscheck(ccall( (:cusolverSpSetStream, libcusolver), cusolverStatus_t, (cusolverSpHandle_t, cudaStream_t), handle, streamId))
end
function cusolverSpGetStream(handle, streamId)
  statuscheck(ccall( (:cusolverSpGetStream, libcusolver), cusolverStatus_t, (cusolverSpHandle_t, Ptr{cudaStream_t}), handle, streamId))
end
function cusolverSpCreateCsrqrInfo(info)
  statuscheck(ccall( (:cusolverSpCreateCsrqrInfo, libcusolver), cusolverStatus_t, (Ptr{csrqrInfo_t},), info))
end
function cusolverSpDestroyCsrqrInfo(info)
  statuscheck(ccall( (:cusolverDestroyCsrqrInfo, libcusolver), cusolverStatus_t, (csrqrInfo_t,), info))
end
function cusolverRfCreate(handle)
  statuscheck(ccall( (:cusolverRfCreate, libcusolver), cusolverStatus_t, (Ptr{cusolverRfHandle_t},), handle))
end
function cusolverRfDestroy(handle)
  statuscheck(ccall( (:cusolverRfDestroy, libcusolver), cusolverStatus_t, (cusolverRfHandle_t,), handle))
end
function cusolverRfSetStream(handle, streamId)
  statuscheck(ccall( (:cusolverRfSetStream, libcusolver), cusolverStatus_t, (cusolverRfHandle_t, cudaStream_t), handle, streamId))
end
function cusolverRfGetStream(handle, streamId)
  statuscheck(ccall( (:cusolverRfGetStream, libcusolver), cusolverStatus_t, (cusolverRfHandle_t, Ptr{cudaStream_t}), handle, streamId))
end
