# Version management

"""
Returns the CUDA driver version as a VersionNumber.
"""
function version()
    version_ref = Ref{Cint}()
    @apicall(:cuDriverGetVersion, (Ptr{Cint},), version_ref)

    major = version_ref[] ÷ 1000
    minor = mod(version_ref[], 100) ÷ 10

    return VersionNumber(major, minor)
end
