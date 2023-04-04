# Version management

# TODO: get rid of release getters

"""
    driver_version()

Returns the latest version of CUDA supported by the loaded driver.
"""
function driver_version()
    version_ref = Ref{Cint}()
    cuDriverGetVersion(version_ref)
    major, ver = divrem(version_ref[], 1000)
    minor, patch = divrem(ver, 10)
    return VersionNumber(major, minor, patch)
end

"""
    system_driver_version()

Returns the latest version of CUDA supported by the original system driver, or
`nothing` if the driver was not upgraded.
"""
function system_driver_version()
    # on unsupported platforms, CUDA_Driver_jll's init function does not run
    if !isdefined(CUDA_Driver_jll, :libcuda_original_version)
        return nothing
    end
    CUDA_Driver_jll.libcuda_original_version
end

"""
    runtime_version()

Returns the CUDA Runtime version.
"""
function runtime_version()
    version_ref = Ref{Cint}()
    @check @ccall libcudart.cudaRuntimeGetVersion(version_ref::Ptr{Cint})::CUresult
    major, ver = divrem(version_ref[], 1000)
    minor, patch = divrem(ver, 10)
    return VersionNumber(major, minor, patch)
end

"""
    set_runtime_version!([version])

Sets the CUDA Runtime version preference to `version`. This can be a version number, in
which case such a versioned artifact will be attempted to be used; or "local" for using a
runtime from the local system. Invoke this function without an argument to reset the
preference, in which case CUDA.jl will use the most recent compatible runtime available.
"""
function set_runtime_version!(version)
    if version isa VersionNumber
        version = "$(version.major).$(version.minor)"
    end
    Preferences.set_preferences!(CUDA_Runtime_jll, "version" => version; force=true)
    @info "Set CUDA Runtime version preference to $version, please re-start Julia for this to take effect."
    if VERSION <= v"1.6.5" || VERSION == v"1.7.0"
        @warn """Due to a bug in Julia (until 1.6.5 and 1.7.1) your environment needs to directly include CUDA_Runtime_jll for this to work."""
    end
end
function set_runtime_version!()
    Preferences.delete_preferences!(CUDA_Runtime_jll, "version"; force=true)
    @info "Reset CUDA Runtime version preference, please re-start Julia for this to take effect."
    if VERSION <= v"1.6.5" || VERSION == v"1.7.0"
        @warn """Due to a bug in Julia (until 1.6.5 and 1.7.1) your environment needs to directly include CUDA_Runtime_jll for this to work."""
    end
end
