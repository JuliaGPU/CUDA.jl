@public version, driver_version, cuda_driver_version

"""
    NVML.version() -> VersionNumber

Return the version of the NVML API. The major version is the CUDA major version the
driver supports, while the driver part of the version string (e.g. `560.35.03` in
`12.560.35.03`) is stored in the build suffix: `v"12.0.0+560.35.3"`. Minor versions are
not reported, so compare against `NVML.driver_version()` or
`NVML.cuda_driver_version()` to check for features added in a minor release. Wraps
`nvmlSystemGetNVMLVersion`.
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

"""
    NVML.driver_version() -> VersionNumber

The version of the installed NVIDIA driver, e.g. `v"560.35.3"` for driver 560.35.03. Wraps
`nvmlSystemGetDriverVersion`. Unlike `CUDACore.driver_version()`, this is not the CUDA
version the driver supports: use [`NVML.cuda_driver_version`](@ref) for that.
"""
function driver_version()
    buf = Vector{Cchar}(undef, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE)
    nvmlSystemGetDriverVersion(pointer(buf), length(buf))
    return GC.@preserve buf VersionNumber(unsafe_string(pointer(buf)))
end

"""
    NVML.cuda_driver_version() -> VersionNumber

The latest version of CUDA that the system's CUDA driver library supports, e.g.
`v"12.6.0"`. Wraps `nvmlSystemGetCudaDriverVersion_v2`, which asks the `libcuda` library
that NVML finds. This usually equals `CUDACore.driver_version()`, but differs when CUDA.jl
loaded another driver library, such as a forward-compatible one.
"""
function cuda_driver_version()
    ref = Ref{Cint}()
    nvmlSystemGetCudaDriverVersion_v2(ref)
    major, ver = divrem(ref[], 1000)
    minor, patch = divrem(ver, 10)
    return VersionNumber(major, minor, patch)
end
