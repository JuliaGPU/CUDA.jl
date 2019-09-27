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
