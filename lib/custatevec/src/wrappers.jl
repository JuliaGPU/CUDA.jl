function custatevecCreate()
  handle = Ref{custatevecHandle_t}()
  custatevecCreate(handle)
  return handle[]
end

function version()
  ver = custatevecGetVersion()
  major, ver = divrem(ver, 1000)
  minor, patch = divrem(ver, 100)

  VersionNumber(major, minor, patch)
end
