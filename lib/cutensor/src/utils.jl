# wrappers of low-level functionality

"""
    cuTENSOR.version() -> VersionNumber

The version of the cuTENSOR library in use. This is the library's own version, not that
of the CUDA toolkit.
"""
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

"""
    cuTENSOR.read_cache!(filename::String)

Load kernels that [`write_cache!`](@ref) saved to `filename` into the kernel cache of the
current task's cuTENSOR handle, so that plans made with `jit=JIT_MODE_DEFAULT` don't have to
compile them again. Wraps `cutensorReadKernelCacheFromFile`.
"""
function read_cache!(filename::String)
  cutensorReadKernelCacheFromFile(handle(), filename)
end

"""
    cuTENSOR.write_cache!(filename::String)

Save the kernels that cuTENSOR compiled for plans made with `jit=JIT_MODE_DEFAULT`, from
the kernel cache of the current task's cuTENSOR handle, to `filename`, for
[`read_cache!`](@ref) to load in a later session. Throws a [`CUTENSORError`](@ref) when the
GPU does not support compiling kernels. Wraps `cutensorWriteKernelCacheToFile`.
"""
function write_cache!(filename::String)
  cutensorWriteKernelCacheToFile(handle(), filename)
end
