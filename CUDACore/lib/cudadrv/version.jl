# Version management

@public driver_version, runtime_version, set_runtime_version!, reset_runtime_version!, compiler_version, is_tegra

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
    runtime_version()

Returns the CUDA Runtime version.

On Linux, this calls `cudaRuntimeGetVersion`, which returns a compile-time constant baked
into the loaded `libcudart`. On Windows we instead read the cudart DLL's PE file version
directly, because NVIDIA's `cudaRuntimeGetVersion` there delegates to a driver-bundled
"hybrid" runtime and reports the driver's version rather than the loaded toolkit's.
"""
function runtime_version()
    @memoize begin
        @static if Sys.iswindows()
            # Read the cudart DLL's PE VS_FIXEDFILEINFO via Win32. NVIDIA encodes the
            # toolkit major.minor in the low 16 bits of dwFileVersionLS as a decimal MMm0
            # (e.g. 13010 → 13.1.0).
            path = Libdl.dlpath(libcudart)
            hver = Libdl.dlopen("version.dll")
            GetFileVersionInfoSizeA = Libdl.dlsym(hver, :GetFileVersionInfoSizeA)
            GetFileVersionInfoA     = Libdl.dlsym(hver, :GetFileVersionInfoA)
            VerQueryValueA          = Libdl.dlsym(hver, :VerQueryValueA)

            handle_ref = Ref{Culong}(0)
            sz = ccall(GetFileVersionInfoSizeA, Culong, (Cstring, Ptr{Culong}), path, handle_ref)
            sz == 0 && error("GetFileVersionInfoSizeA returned 0 for $path")
            buf = Vector{UInt8}(undef, sz)
            ok = ccall(GetFileVersionInfoA, Cint, (Cstring, Culong, Culong, Ptr{UInt8}), path, 0, sz, buf)
            ok == 0 && error("GetFileVersionInfoA failed for $path")

            info_ptr = Ref{Ptr{UInt8}}(C_NULL)
            len_ref  = Ref{Cuint}(0)
            ok = ccall(VerQueryValueA, Cint, (Ptr{UInt8}, Cstring, Ptr{Ptr{UInt8}}, Ptr{Cuint}),
                       buf, "\\", info_ptr, len_ref)
            ok == 0 && error("VerQueryValueA failed for $path")

            # VS_FIXEDFILEINFO: dwSignature, dwStrucVersion, dwFileVersionMS, dwFileVersionLS, ...
            dwFileVersionLS = unsafe_load(reinterpret(Ptr{UInt32}, info_ptr[]), 4)
            encoded = Int(dwFileVersionLS & 0xffff)
            major, ver = divrem(encoded, 1000)
            minor, patch = divrem(ver, 10)
            VersionNumber(major, minor, patch)
        else
            v = Ref{Cint}()
            check() do
                @ccall libcudart.cudaRuntimeGetVersion(v::Ptr{Cint})::CUresult
            end
            major, ver = divrem(v[], 1000)
            minor, patch = divrem(ver, 10)
            VersionNumber(major, minor, patch)
        end
    end::VersionNumber
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
[`CUDACore.reset_runtime_version!`](@ref) instead.
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
local project, use [`CUDACore.set_runtime_version!`](@ref) with no arguments.
"""
function reset_runtime_version!()
    Preferences.delete_preferences!(CUDA_Runtime_jll, "version"; force=true)
    Preferences.delete_preferences!(CUDA_Runtime_jll, "local"; force=true)
    @info "Reset CUDA.jl toolkit preference, please re-start Julia for this to take effect."
end

"""
    compiler_version()

Returns the CUDA toolkit version that is used to provide the CUDA compiler (`ptxas`) and
other tools. This is versioned separately from the CUDA Runtime, in order to ensure
compatibility with the driver, and make sure we use the latest compatible version regardless
of the selected runtime.

Derived by parsing `ptxas --version`.
"""
function compiler_version()
    @memoize begin
        output = readchomp(`$(CUDA_Compiler.ptxas()) --version`)
        m = match(r"release (\d+)\.(\d+),\s*V(\d+)\.(\d+)\.(\d+)", output)
        m === nothing && error("Could not parse `ptxas --version` output: $output")
        VersionNumber(parse(Int, m.captures[3]),
                      parse(Int, m.captures[4]),
                      parse(Int, m.captures[5]))
    end::VersionNumber
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
