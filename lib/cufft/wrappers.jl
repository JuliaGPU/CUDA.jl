# wrappers of low-level functionality

function cufftGetProperty(property::libraryPropertyType)
  value_ref = Ref{Cint}()
  cufftGetProperty(property, value_ref)
  value_ref[]
end

version() = VersionNumber(cufftGetProperty(CUDA.MAJOR_VERSION),
                          cufftGetProperty(CUDA.MINOR_VERSION),
                          cufftGetProperty(CUDA.PATCH_LEVEL))
