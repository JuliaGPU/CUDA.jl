# wrappers of low-level functionality

function cufftGetProperty(property::CUDAapi.libraryPropertyType)
  value_ref = Ref{Cint}()
  cufftGetProperty(property, value_ref)
  value_ref[]
end

version() = VersionNumber(cufftGetProperty(CUDAapi.MAJOR_VERSION),
                          cufftGetProperty(CUDAapi.MINOR_VERSION),
                          cufftGetProperty(CUDAapi.PATCH_LEVEL))
