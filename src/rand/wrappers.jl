# wrappers of low-level functionality

function curandGetProperty(property::CUDAapi.libraryPropertyType)
  value_ref = Ref{Cint}()
  curandGetProperty(property, value_ref)
  value_ref[]
end

version() = VersionNumber(curandGetProperty(CUDAapi.MAJOR_VERSION),
                          curandGetProperty(CUDAapi.MINOR_VERSION),
                          curandGetProperty(CUDAapi.PATCH_LEVEL))

macro allocates(ex)
  quote
    CuArrays.extalloc(check=err->err.code == CURAND_STATUS_ALLOCATION_FAILED) do
      $(esc(ex))
    end
  end
end
