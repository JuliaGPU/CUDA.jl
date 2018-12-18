# Julia wrapper for header: /usr/local/cuda/include/cusolver.h

#helper functions
function cusolverDnCreate()
  handle = Ref{cusolverDnHandle_t}()
  @check ccall((:cusolverDnCreate, libcusolver),
               cusolverStatus_t,
               (Ptr{cusolverDnHandle_t},),
               handle)
  return handle[]
end
function cusolverDnDestroy(handle)
  @check ccall((:cusolverDnDestroy, libcusolver),
               cusolverStatus_t,
               (cusolverDnHandle_t,),
               handle)
end
function cusolverDnSetStream(handle, streamId)
  @check ccall((:cusolverDnSetStream, libcusolver),
               cusolverStatus_t,
               (cusolverDnHandle_t, CuStream_t),
               handle, streamId)
end
function cusolverDnGetStream(handle, streamId)
  @check ccall((:cusolverDnGetStream, libcusolver),
               cusolverStatus_t,
               (cusolverDnHandle_t, Ptr{CuStream_t}),
               handle, streamId)
end
function cusolverSpCreate()
  handle = Ref{cusolverSpHandle_t}()
  @check ccall((:cusolverSpCreate, libcusolver),
               cusolverStatus_t,
               (Ptr{cusolverSpHandle_t},),
               handle)
  return handle[]
end
function cusolverSpDestroy(handle)
  @check ccall((:cusolverSpDestroy, libcusolver),
               cusolverStatus_t,
               (cusolverSpHandle_t,),
               handle)
end
function cusolverSpSetStream(handle, streamId)
  @check ccall((:cusolverSpSetStream, libcusolver),
               cusolverStatus_t,
               (cusolverSpHandle_t, CuStream_t),
               handle, streamId)
end
function cusolverSpGetStream(handle, streamId)
  @check ccall((:cusolverSpGetStream, libcusolver),
               cusolverStatus_t,
               (cusolverSpHandle_t, Ptr{CuStream_t}),
               handle, streamId)
end
function cusolverSpCreateCsrqrInfo(info)
  @check ccall((:cusolverSpCreateCsrqrInfo, libcusolver),
               cusolverStatus_t,
               (Ptr{csrqrInfo_t},),
               info)
end
function cusolverSpDestroyCsrqrInfo(info)
  @check ccall((:cusolverDestroyCsrqrInfo, libcusolver),
               cusolverStatus_t,
               (csrqrInfo_t,),
               info)
end
function cusolverDnCreateGesvdjInfo(info)
  @check ccall((:cusolverDnCreateGesvdjInfo, libcusolver),
               cusolverStatus_t,
               (Ptr{gesvdjInfo_t},),
               info)
end
function cusolverDnDestroyGesvdjInfo(info)
  @check ccall((:cusolverDnDestroyGesvdjInfo, libcusolver),
               cusolverStatus_t,
               (gesvdjInfo_t,),
               info)
end
function cusolverDnXgesvdjSetTolerance(info, tolerance)
  @check ccall((:cusolverDnXgesvdjSetTolerance, libcusolver),
               cusolverStatus_t,
               (gesvdjInfo_t, Float64),
               info, Float64(tolerance))
end
function cusolverDnXgesvdjSetMaxSweeps(info, max_sweeps)
  @check ccall((:cusolverDnXgesvdjSetMaxSweeps, libcusolver),
               cusolverStatus_t,
               (gesvdjInfo_t, Cint),
               info, Cint(max_sweeps))
end
function cusolverDnCreateSyevjInfo(info)
  @check ccall((:cusolverDnCreateSyevjInfo, libcusolver),
               cusolverStatus_t,
               (Ptr{syevjInfo_t},),
               info)
end
function cusolverDnDestroySyevjInfo(info)
  @check ccall((:cusolverDnDestroySyevjInfo, libcusolver),
               cusolverStatus_t,
               (syevjInfo_t,),
               info)
end
function cusolverDnXsyevjSetTolerance(info, tolerance)
  @check ccall((:cusolverDnXsyevjSetTolerance, libcusolver),
               cusolverStatus_t,
               (syevjInfo_t, Float64),
               info, Float64(tolerance))
end
function cusolverDnXsyevjSetMaxSweeps(info, max_sweeps)
  @check ccall((:cusolverDnXsyevjSetMaxSweeps, libcusolver),
               cusolverStatus_t,
               (syevjInfo_t, Cint),
               info, Cint(max_sweeps))
end
function cusolverRfCreate(handle)
  @check ccall((:cusolverRfCreate, libcusolver),
               cusolverStatus_t,
               (Ptr{cusolverRfHandle_t},),
               handle)
end
function cusolverRfDestroy(handle)
  @check ccall((:cusolverRfDestroy, libcusolver),
               cusolverStatus_t,
               (cusolverRfHandle_t,),
               handle)
end
function cusolverRfSetStream(handle, streamId)
  @check ccall((:cusolverRfSetStream, libcusolver),
               cusolverStatus_t,
               (cusolverRfHandle_t, CuStream_t),
               handle, streamId)
end
function cusolverRfGetStream(handle, streamId)
  @check ccall((:cusolverRfGetStream, libcusolver),
               cusolverStatus_t,
               (cusolverRfHandle_t, Ptr{CuStream_t}),
               handle, streamId)
end

function cusolverGetProperty(property::CUDAapi.libraryPropertyType)
  value_ref = Ref{Cint}()
  @check ccall((:cusolverGetProperty, libcusolver),
               cusolverStatus_t,
               (Cint, Ptr{Cint}),
               property, value_ref)
  value_ref[]
end
