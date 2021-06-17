function cudnnCreate()
    handle_ref = Ref{cudnnHandle_t}()
    @check unsafe_cudnnCreate(handle_ref) CUDNN_STATUS_NOT_INITIALIZED CUDNN_STATUS_INTERNAL_ERROR
    return handle_ref[]
end

function cudnnGetProperty(property::CUDA.libraryPropertyType)
  value_ref = Ref{Cint}()
  cudnnGetProperty(property, value_ref)
  value_ref[]
end

function version()
  ver = cudnnGetVersion()
  major, ver = divrem(ver, 1000)
  minor, patch = divrem(ver, 10)

  VersionNumber(major, minor, patch)
end

function cuda_version()
  ver = cudnnGetCudartVersion()
  major, ver = divrem(ver, 1000)
  minor, patch = divrem(ver, 10)

  VersionNumber(major, minor, patch)
end
