# Version management

"""
    version()

Returns the latest version of CUDA supported by the driver.
"""
function version()
    version_ref = Ref{Cint}()
    cuDriverGetVersion(version_ref)
    major, ver = divrem(version_ref[], 1000)
    minor, patch = divrem(ver, 10)
    return VersionNumber(major, minor, patch)
end

"""
    release()

Returns the CUDA release part of the version as returned by [`version`](@ref).
"""
release() = VersionNumber(version().major, version().minor)

"""
    runtime_version()

    Returns the CUDA Runtime version.
"""
function runtime_version()
    version_ref = Ref{Cint}()
    @ccall libcudart().cudaRuntimeGetVersion(version_ref::Ptr{Cint})::CUresult
    major, ver = divrem(version_ref[], 1000)
    minor, patch = divrem(ver, 10)
    return VersionNumber(major, minor, patch)
end
