function version()
    buf = Vector{Cchar}(undef, NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE)
    nvmlSystemGetNVMLVersion(pointer(buf), length(buf))

    # the version string is too long for Julia to handle, e.g. 11.450.36.06,
    # so split off the driver part into the build suffix
    ver = unsafe_string(pointer(buf))
    parts = parse.(Int, split(ver, '.'))
    return VersionNumber(parts[1], 0, 0, (), Tuple(parts[2:end]))
end

function driver_version()
    buf = Vector{Cchar}(undef, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE)
    nvmlSystemGetDriverVersion(pointer(buf), length(buf))
    return VersionNumber(unsafe_string(pointer(buf)))
end

function cuda_driver_version()
    ref = Ref{Cint}()
    nvmlSystemGetCudaDriverVersion_v2(ref)
    major, minor = divrem(ref[], 1000)
    return VersionNumber(major, minor)
end
