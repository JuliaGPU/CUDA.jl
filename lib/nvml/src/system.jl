"""
    NVML.version()

Return the version of the NVML API. The major version is the CUDA major version the
driver supports, while the driver part of the version string (e.g. `560.35.03` in
`12.560.35.03`) is stored in the build suffix: `v"12.0.0+560.35.3"`. Minor versions are
not reported, so compare against `NVML.driver_version()` or
`NVML.cuda_driver_version()` to check for features added in a minor release.
"""
function version()
    buf = Vector{Cchar}(undef, NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE)
    nvmlSystemGetNVMLVersion(pointer(buf), length(buf))

    # the version string is too long for Julia to handle, e.g. 11.450.36.06,
    # so split off the driver part into the build suffix
    ver = GC.@preserve buf unsafe_string(pointer(buf))
    parts = parse.(Int, split(ver, '.'))
    return VersionNumber(parts[1], 0, 0, (), Tuple(parts[2:end]))
end

function driver_version()
    buf = Vector{Cchar}(undef, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE)
    nvmlSystemGetDriverVersion(pointer(buf), length(buf))
    return GC.@preserve buf VersionNumber(unsafe_string(pointer(buf)))
end

function cuda_driver_version()
    ref = Ref{Cint}()
    nvmlSystemGetCudaDriverVersion_v2(ref)
    major, ver = divrem(ref[], 1000)
    minor, patch = divrem(ver, 10)
    return VersionNumber(major, minor, patch)
end
