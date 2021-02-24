# wrappers of low-level functionality

function curandCreateGenerator(typ)
  handle_ref = Ref{curandGenerator_t}()
  @check unsafe_curandCreateGenerator(handle_ref, typ) CURAND_STATUS_INITIALIZATION_FAILED
  handle_ref[]
end

@memoize function curandGetProperty(property::libraryPropertyType)
  value_ref = Ref{Cint}()
  curandGetProperty(property, value_ref)
  value_ref[]
end

@memoize version() = VersionNumber(curandGetProperty(CUDA.MAJOR_VERSION),
                                   curandGetProperty(CUDA.MINOR_VERSION),
                                   curandGetProperty(CUDA.PATCH_LEVEL))
