# wrappers of low-level functionality

function curandCreateGenerator(typ)
  handle_ref = Ref{curandGenerator_t}()
  res = @retry_reclaim err->isequal(err, CURAND_STATUS_ALLOCATION_FAILED) ||
                            isequal(err, CURAND_STATUS_INITIALIZATION_FAILED) begin
    unsafe_curandCreateGenerator(handle_ref, typ)
  end
  if res != CURAND_STATUS_SUCCESS
    throw_api_error(res)
  end
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
