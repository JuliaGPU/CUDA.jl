# wrappers of low-level functionality

function curandCreateGenerator(typ)
  handle_ref = Ref{curandGenerator_t}()
  check(CURAND_STATUS_INITIALIZATION_FAILED) do
    unsafe_curandCreateGenerator(handle_ref, typ)
  end
  handle_ref[]
end

function curandGetProperty(property::libraryPropertyType)
  value_ref = Ref{Cint}()
  curandGetProperty(property, value_ref)
  value_ref[]
end

version() = VersionNumber(curandGetProperty(CUDA.MAJOR_VERSION),
                          curandGetProperty(CUDA.MINOR_VERSION),
                          curandGetProperty(CUDA.PATCH_LEVEL))
