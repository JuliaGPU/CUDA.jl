function cudnnCreate()
    handle = Ref{cudnnHandle_t}()
    cudnnCreate(handle)
    return handle[]
end

function cudnnGetProperty(property::CUDAapi.libraryPropertyType)
  value_ref = Ref{Cint}()
  cudnnGetProperty(property, value_ref)
  value_ref[]
end

version() = VersionNumber(cudnnGetProperty(CUDAapi.MAJOR_VERSION),
                          cudnnGetProperty(CUDAapi.MINOR_VERSION),
                          cudnnGetProperty(CUDAapi.PATCH_LEVEL))

function cuda_version()
  ver = cudnnGetCudartVersion()
  major, ver = divrem(ver, 1000)
  minor, patch = divrem(ver, 10)

  VersionNumber(major, minor, patch)
end
