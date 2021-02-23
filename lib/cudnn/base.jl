function cudnnCreate()
    handle_ref = Ref{cudnnHandle_t}()
    @check unsafe_cudnnCreate(handle_ref) CUDNN_STATUS_NOT_INITIALIZED CUDNN_STATUS_INTERNAL_ERROR
    return handle_ref[]
end

@memoize function cudnnGetProperty(property::CUDA.libraryPropertyType)
  value_ref = Ref{Cint}()
  cudnnGetProperty(property, value_ref)
  value_ref[]
end

@memoize function version()
  ver = cudnnGetVersion()
  major, ver = divrem(ver, 1000)
  minor, patch = divrem(ver, 10)

  VersionNumber(major, minor, patch)
end

@memoize function cuda_version()
  ver = cudnnGetCudartVersion()
  major, ver = divrem(ver, 1000)
  minor, patch = divrem(ver, 10)

  VersionNumber(major, minor, patch)
end
