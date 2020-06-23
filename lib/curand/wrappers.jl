# wrappers of low-level functionality

function curandGetProperty(property::libraryPropertyType)
  value_ref = Ref{Cint}()
  curandGetProperty(property, value_ref)
  value_ref[]
end

version() = VersionNumber(curandGetProperty(CUDA.MAJOR_VERSION),
                          curandGetProperty(CUDA.MINOR_VERSION),
                          curandGetProperty(CUDA.PATCH_LEVEL))
