# Version management

# because of this API call being used so frequently, we use a manual cache set in __init__
# (@memoize's lazy/thread-safe initialization is too expensive for this purpose)
const _driver_version = Ref{VersionNumber}()
function set_driver_version()
    version_ref = Ref{Cint}()
    cuDriverGetVersion(version_ref)
    major, ver = divrem(version_ref[], 1000)
    minor, patch = divrem(ver, 10)
    _driver_version[] = VersionNumber(major, minor, patch)
end

"""
    driver_version()

Returns the latest version of CUDA supported by the loaded driver.
"""
function driver_version()
    assume(isassigned(_driver_version))
    _driver_version[]
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
    CUDA.set_runtime_version!([version::VersionNumber]; [local_toolkit::Bool])

Configures the active project to use a specific CUDA toolkit version from a specific source.

If `local_toolkit` is set, the CUDA toolkit will be used from the local system, otherwise it
will be downloaded from an artifact source. In the case of a local toolkit, `version`
informs CUDA.jl which version that is (this may be useful if auto-detection fails). In the
case of artifact sources, `version` controls which version will be downloaded and used.

When not specifying either the `version` or the `local_toolkit` argument, the default
behavior will be used, which is to use the most recent compatible runtime available from an
artifact source. Note that this will override any Preferences that may be configured in a
higher-up depot; to clear preferences nondestructively, use
[`CUDA.reset_runtime_version!`](@ref) instead.
"""
function set_runtime_version!(version::Union{Nothing,VersionNumber}=nothing;
                              local_toolkit::Union{Nothing,Bool}=nothing)
    # store stringified properties
    let version = isnothing(version) ? nothing : "$(version.major).$(version.minor)"
        Preferences.set_preferences!(CUDA_Runtime_jll, "version" => version; force=true)
    end
    let local_toolkit = isnothing(local_toolkit) ? nothing : string(local_toolkit)
        Preferences.set_preferences!(CUDA_Runtime_jll, "local" => local_toolkit; force=true)
    end

    io = IOBuffer()
    print(io, "Configure the active project to use ")
    if version !== nothing
        print(io, "CUDA $(version.major).$(version.minor)")
    else
        print(io, "the default CUDA")
    end
    if local_toolkit !== nothing
        print(io, local_toolkit ? " from the local system" : " from artifact sources")
    end
    print(io, "; please re-start Julia for this to take effect.")
    @info String(take!(io))
end

"""
    CUDA.reset_runtime_version!()

Resets the CUDA version preferences in the active project to the default, which is to use
the most recent compatible runtime available from an artifact source, unless a higher-up
depot has configured a different preference. To force use of the default behavior for the
local project, use [`CUDA.set_runtime_version!`](@ref) with no arguments.
"""
function reset_runtime_version!()
    Preferences.delete_preferences!(CUDA_Runtime_jll, "version"; force=true)
    Preferences.delete_preferences!(CUDA_Runtime_jll, "local"; force=true)
    @info "Reset CUDA.jl toolkit preference, please re-start Julia for this to take effect."
end


## helpers

function is_tegra()
    if !Sys.islinux()
        return false
    end
    if isfile("/etc/nv_tegra_release")
        return true
    end
    if isfile("/proc/device-tree/compatible") &&
        contains(read("/proc/device-tree/compatible", String), "tegra")
        return true
    end
    return false
end
