# Version management

"""
    version()

Returns the CUDA version as reported by the driver.
"""
function version()
    version_ref = Ref{Cint}()
    cuDriverGetVersion(version_ref)
    major, minor = divrem(version_ref[], 1000)
    return VersionNumber(major, minor)
end

"""
    release()

Returns the CUDA release part of the version as returned by [`version`](@ref).
"""
release() = VersionNumber(version().major, version().minor)
