# wrappers of low-level functionality

function curandGetProperty(property::CUDAapi.libraryPropertyType)
  value_ref = Ref{Cint}()
  curandGetProperty(property, value_ref)
  value_ref[]
end

version() = VersionNumber(curandGetProperty(CUDAapi.MAJOR_VERSION),
                          curandGetProperty(CUDAapi.MINOR_VERSION),
                          curandGetProperty(CUDAapi.PATCH_LEVEL))
