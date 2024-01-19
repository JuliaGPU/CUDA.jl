# wrappers of low-level functionality

function version()
    ver = cutensorGetVersion()
    major, ver = divrem(ver, 10000)
    minor, patch = divrem(ver, 100)

    VersionNumber(major, minor, patch)
end

function cuda_version()
  ver = cutensorGetCudartVersion()
  major, ver = divrem(ver, 1000)
  minor, patch = divrem(ver, 10)

  VersionNumber(major, minor, patch)
end

function cutensorCreate()
  handle_ref = Ref{cutensorHandle_t}()
  cutensorCreate(handle_ref)
  handle_ref[]
end

function read_cache!(filename::String)
  cutensorReadKernelCacheFromFile(handle(), filename)
end

function write_cache!(filename::String)
  cutensorWriteKernelCacheToFile(handle(), filename)
end
