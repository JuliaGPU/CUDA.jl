# wrappers of low-level functionality

function curandGetProperty(property::libraryPropertyType)
  value_ref = Ref{Cint}()
  curandGetProperty(property, value_ref)
  value_ref[]
end

version() = VersionNumber(curandGetProperty(MAJOR_VERSION),
                          curandGetProperty(MINOR_VERSION),
                          curandGetProperty(PATCH_LEVEL))
