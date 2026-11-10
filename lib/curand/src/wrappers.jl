# wrappers of low-level functionality

function curandCreateGenerator(typ)
  handle_ref = Ref{curandGenerator_t}()
  curandCreateGenerator(handle_ref, typ)
  handle_ref[]
end

function curandGetProperty(property::libraryPropertyType)
  value_ref = Ref{Cint}()
  curandGetProperty(property, value_ref)
  value_ref[]
end

"""
    cuRAND.version() -> VersionNumber

The version of the cuRAND library in use. This is the library's own version, which differs
from that of the CUDA toolkit it ships with.
"""
version() = VersionNumber(curandGetProperty(CUDACore.MAJOR_VERSION),
                          curandGetProperty(CUDACore.MINOR_VERSION),
                          curandGetProperty(CUDACore.PATCH_LEVEL))
