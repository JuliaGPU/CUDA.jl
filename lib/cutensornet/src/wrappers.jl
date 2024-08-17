function cutensornetCreate()
    handle = Ref{cutensornetHandle_t}()
    cutensornetCreate(handle)
    handle[]
end

function version()
  ver = cutensornetGetVersion()
  major, ver = divrem(ver, 10000)
  minor, patch = divrem(ver, 100)

  VersionNumber(major, minor, patch)
end

function cuda_version()
  ver = cutensornetGetCudartVersion()
  major, ver = divrem(ver, 1000)
  minor, patch = divrem(ver, 10)

  VersionNumber(major, minor, patch)
end
