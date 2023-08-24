# Version management

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
    check() do
        @ccall libcudart.cudaRuntimeGetVersion(version_ref::Ptr{Cint})::CUresult
    end
    major, ver = divrem(version_ref[], 1000)
    minor, patch = divrem(ver, 10)
    return VersionNumber(major, minor, patch)
end

"""
    set_runtime_version!([version::VersionNumber]; local_toolkit=false)

Configures CUDA.jl to use a specific CUDA toolkit version from a specific source.

If `local_toolkit` is set, the CUDA toolkit will be used from the local system, otherwise
it will be downloaded from an artifact source. In the case of a local toolkit, `version`
informs CUDA.jl which version that is (this may be useful if auto-detection fails). In
the case of artifact sources, `version` controls which version will be downloaded and used.

See also: [`reset_runtime_version!`](@ref).
"""
function set_runtime_version!(version::Union{Nothing,VersionNumber}=nothing;
                              local_toolkit::Bool=false)
    if version !== nothing
        Preferences.set_preferences!(CUDA_Runtime_jll, "version" => "$(version.major).$(version.minor)"; force=true)
    else
        Preferences.delete_preferences!(CUDA_Runtime_jll, "version"; force=true)
    end
    if local_toolkit
        Preferences.set_preferences!(CUDA_Runtime_jll, "local" => "true"; force=true)
    else
        # the default is "false"
        Preferences.delete_preferences!(CUDA_Runtime_jll, "local"; force=true)
    end
    @info "Set CUDA.jl toolkit preference to use $(version === nothing ? "CUDA" : "CUDA $version") from $(local_toolkit ? "the local system" : "artifact sources"), please re-start Julia for this to take effect."
end

"""
    reset_runtime_version!()

Resets the CUDA Runtime version preference to the default, which is to use the most recent
compatible runtime available from an artifact source.

See also: [`set_runtime_version!`](@ref).
"""
function reset_runtime_version!()
    Preferences.delete_preferences!(CUDA_Runtime_jll, "version"; force=true)
    Preferences.delete_preferences!(CUDA_Runtime_jll, "local"; force=true)
    @info "Reset CUDA.jl toolkit preference, please re-start Julia for this to take effect."
end


## helpers

is_tegra() = Sys.islinux() && isfile("/etc/nv_tegra_release")
