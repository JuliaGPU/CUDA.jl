function custatevecCreate()
  handle = Ref{custatevecHandle_t}()
  custatevecCreate(handle)
  return handle[]
end

"""
    cuStateVec.version() -> VersionNumber

The version of the cuStateVec library in use, not that of the CUDA toolkit.
"""
function version()
  ver = custatevecGetVersion()
  major, ver = divrem(ver, 10000)
  minor, patch = divrem(ver, 100)

  VersionNumber(major, minor, patch)
end
