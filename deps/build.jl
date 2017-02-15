# Discover the CUDA library

using Compat

const library = Libdl.find_library(is_windows() ? "nvcuda.dll" : "libcuda")
if library == ""
    error("Could not find CUDA library -- is the CUDA driver installed?")
end

# save additional info
library_path = Libdl.dlpath(library)
vendor = "NVIDIA"

# find the library version
# NOTE: should be kept in sync with src/version.jl::version()
version_ref = Ref{Cint}()
status = ccall((:cuDriverGetVersion, library), Cint, (Ptr{Cint},), version_ref)
if status != 0
    error("Could not get CUDA library version")
end
major = version_ref[] ÷ 1000
minor = mod(version_ref[], 100) ÷ 10
version = VersionNumber(major, minor)

# write ext.jl
open(joinpath(@__DIR__, "ext.jl"), "w") do fh
    write(fh, """
        const libcuda = "$library"
        const libcuda_path = "$library_path"
        const libcuda_version = v"$version"
        const libcuda_vendor = "$vendor"
        """)
end
