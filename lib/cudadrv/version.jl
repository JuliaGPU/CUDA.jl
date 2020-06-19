# Version management

"""
    version()

Returns the CUDA version as reported by the driver.
"""
function version()
    version_ref = Ref{Cint}()
    cuDriverGetVersion(version_ref)

    major = version_ref[] ÷ 1000
    minor = Base.mod(version_ref[], 100) ÷ 10

    return VersionNumber(major, minor)
end

"""
    release()

Returns the CUDA release part of the version as returned by [`version`](@ref).
"""
release() = VersionNumber(version().major, version().minor)
