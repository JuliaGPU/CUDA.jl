function cutensornetCreate()
    handle = Ref{cutensornetHandle_t}()
    cutensornetCreate(handle)
    handle[]
end

"""
    cuTensorNet.version() -> VersionNumber

The version of the cuTensorNet library in use, not that of the CUDA toolkit.
"""
function version()
  ver = cutensornetGetVersion()
  major, ver = divrem(ver, 10000)
  minor, patch = divrem(ver, 100)

  VersionNumber(major, minor, patch)
end

"""
    cuTensorNet.cuda_version() -> VersionNumber

The version of the CUDA runtime that cuTensorNet was built with.
"""
function cuda_version()
  ver = cutensornetGetCudartVersion()
  major, ver = divrem(ver, 1000)
  minor, patch = divrem(ver, 10)

  VersionNumber(major, minor, patch)
end
