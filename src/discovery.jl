export find_library, find_binary,
       find_cuda_library, find_cuda_binary,
       find_toolkit, find_toolkit_version,
       find_libdevice,
       find_host_compiler, find_toolchain

# debug print helpers
source_str(src::String) = "in $src"
source_str(srcs::Vector{String}) = isempty(srcs) ? "anywhere" : "in " * join(srcs, ", ", " or ")
target_str(typ::String, dst::String) = "$typ $dst"
target_str(typ::String, dsts::Vector{String}) = isempty(dsts) ? "no $typ" : "$typ " * join(dsts, ", ", " or ")

function resolve(path)
    if islink(path)
        dir = dirname(path)
        resolve(joinpath(dir, readlink(path)))
    else
        path
    end
end

# return a list of valid directories, resolving symlinks and pruning duplicates
function valid_dirs(dirs)
    map!(resolve, dirs, dirs)
    filter(isdir, unique(dirs))
end


## generic discovery routines

"""
    find_library(names; locations=String[], versions=VersionNumber[], word_size=Sys.WORD_SIZE)

Wrapper for Libdl.find_library, performing a more exhaustive search:

- variants of the library name (including version numbers, platform-specific tags, etc);
- various subdirectories of the `locations` list, and finally system library directories.

Returns the full path to the library.
"""
function find_library(names::Vector{String};
                      locations::Vector{String}=String[],
                      versions::Vector{VersionNumber}=VersionNumber[],
                      word_size::Int=Sys.WORD_SIZE)
    @debug("Request to look $(source_str(locations)) for $(target_str("library", names))")

    # figure out names
    all_names = String[]
    if Compat.Sys.iswindows()
        # priority goes to the `names` argument, as per `Libdl.find_library`
        for name in names
            for version in versions
                append!(all_names, ["$(name)$(word_size)_$(version.major)$(version.minor)",
                                    "$(name)$(word_size)_$(version.major)"])
            end
            # look for unversioned libraries
            append!(all_names, ["$(name)$(word_size)", name])
        end
    else
        all_names = ["lib$name" for name in names]
    end
    # the dual reverse is to put less specific names last,
    # eg. ["lib9.1", "lib9", "lib9.0", "lib9.0"] => ["lib9.1", "lib9.0", "lib9.0"]
    all_names = reverse(unique(reverse(all_names)))

    # figure out locations
    all_locations = String[]
    for location in locations
        push!(all_locations, location)
        if Compat.Sys.iswindows()
            push!(all_locations, joinpath(location, "bin"))
        else
            push!(all_locations, joinpath(location, "lib"))
            if Sys.WORD_SIZE == 64
                push!(all_locations, joinpath(location, "lib64"))
            end
        end
    end

    @debug("Looking $(source_str(locations)) for $(target_str("library", names))")
    name_found = Libdl.find_library(all_names, all_locations)
    if isempty(name_found)
        return nothing
    end

    # find the full path of the library (which Libdl.find_library doesn't guarantee to return)
    path = resolve(Libdl.dlpath(name_found))
    @debug("Found $name_found library at $path")
    return path
end

"""
    find_binary(names; locations=String[])

Similar to `find_library`, performs an exhaustive search for a binary in various
subdirectories of `locations`, and finally PATH.
"""
function find_binary(names::Vector{String};
                     locations::Vector{String}=String[])
    @debug("Request to look $(source_str(locations)) for $(target_str("binary", names))")

    # figure out names
    all_names = String[]
    if Compat.Sys.iswindows()
        all_names = ["$name.exe" for name in names]
    else
        all_names = names
    end

    # figure out locations
    all_locations = String[]
    for location in locations
        push!(all_locations, location)
        push!(all_locations, joinpath(location, "bin"))
    end
    let path = ENV["PATH"]
        dirs = split(path, Compat.Sys.iswindows() ? ';' : ':')
        filter!(path->!isempty(path), dirs)
        append!(all_locations, dirs)
    end

    @debug("Looking $(source_str(all_locations)) for $(target_str("binary", all_names))")
    paths = [joinpath(location, name) for name in all_names, location in all_locations]
    try
        paths = filter(ispath, paths)
    end

    if isempty(paths)
        return nothing
    else
        path = resolve(first(paths))
        @debug("Found binary at $path")
        return path
    end
end


## CUDA-specific discovery routines

# FIXME: CUDA on 32-bit Windows isn't supported

const cuda_names = Dict(
    "cuda"      => Compat.Sys.iswindows() ? ["nvcuda"] : ["cuda"],
    "nvml"      => Compat.Sys.iswindows() ? ["nvml"]   : ["nvidia-ml"]
)

const cuda_versions = Dict(
    "toolkit"   => [v"1.0", v"1.1",
                    v"2.0", v"2.1", v"2.2",
                    v"3.0", v"3.1", v"3.2",
                    v"4.0", v"4.1", v"4.2",
                    v"5.0", v"5.5",
                    v"6.0", v"6.5",
                    v"7.0", v"7.5",
                    v"8.0",
                    v"9.0", v"9.1"],
    "cudnn"     => [v"1.0", v"2.0", v"3.0", v"4.0", v"5.0", v"5.1", v"6.0", v"7.0"]
)

# simplified find_library/find_binary entry-points,
# looking up name aliases and known version numbers
# and passing the (optional) toolkit dirs as locations.
find_cuda_library(name::String, toolkit_dirs::Vector{String}=String[];
                  versions::Vector{VersionNumber}=reverse(get(cuda_versions, name, cuda_versions["toolkit"])),
                  kwargs...) =
    find_library(get(cuda_names, name, [name]);
                 versions=versions,
                 locations=toolkit_dirs,
                 kwargs...)
find_cuda_binary(name::String, toolkit_dirs::Vector{String}=String[]; kwargs...) =
    find_binary(get(cuda_names, name, [name]);
                locations=toolkit_dirs,
                kwargs...)

"""
    find_toolkit()::Vector{String}

Look for directories where (parts of) the CUDA toolkit might be installed. This returns a
(possibly empty) list of paths that can be used as an argument to other discovery functions.

The behavior of this function can be overridden by defining the `CUDA_PATH`, `CUDA_HOME` or
`CUDA_ROOT` environment variables, which should point to the root of the CUDA toolkit.
"""
function find_toolkit()
    # look for environment variables to override discovery
    envvars = ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"]
    envvars_set = filter(var -> haskey(ENV, var), envvars)
    if length(envvars_set) > 0
        envvals = unique(map(var->ENV[var], envvars_set))
        if length(envvals) > 1
            warn("Multiple CUDA environment variables set to different values: $(join(envvars_set, ", ", " and "))")
        end
        @debug("Considering CUDA toolkit at $(envvals...) based on environment variables $(join(envvars_set, ", "))")
        return envvals
    end

    dirs = String[]

    # look for the compiler binary (in the case PATH points to the installation)
    nvcc_path = find_cuda_binary("nvcc")
    if nvcc_path != nothing
        nvcc_dir = dirname(nvcc_path)
        if contains(basename(nvcc_dir), r"^bin(32|64)?$")
            nvcc_dir = dirname(nvcc_dir)
        end
        @debug("Considering CUDA toolkit at $nvcc_dir based on nvcc at $nvcc_path")
        push!(dirs, nvcc_dir)
    end

    # look for the runtime library (in the case LD_LIBRARY_PATH points to the installation)
    libcudart_path = find_cuda_library("cudart")
    if libcudart_path != nothing
        libcudart_dir = dirname(libcudart_path)
        if contains(basename(libcudart_dir), r"^(lib|bin)(32|64)?$")
            libcudart_dir = dirname(libcudart_dir)
        end
        @debug("Considering CUDA toolkit at $libcudart_dir based on libcudart at $libcudart_path")
        push!(dirs, libcudart_dir)
    end

    # look in default installation directories
    if Compat.Sys.iswindows()
        # CUDA versions are installed in separate directories under a single base dir
        program_files = ENV[Sys.WORD_SIZE == 64 ? "ProgramFiles" : "ProgramFiles(x86)" ]
        basedir = joinpath(program_files, "NVIDIA GPU Computing Toolkit", "CUDA")
        @debug("Considering default CUDA installation directory at $basedir")
        if isdir(basedir)
            entries = map(dir -> joinpath(basedir, dir), readdir(basedir))
            reverse!(entries) # we want to search starting from the newest CUDA version
            @debug("Considering CUDA toolkits at $(join(entries, ", ")) based on default installation directory")
            append!(dirs, entries)
        end
    else
        # CUDA versions are installed in unversioned dirs, or suffixed with the version
        basedirs = ["/usr/local/cuda", "/opt/cuda"]
        for dir in basedirs
            append!(dirs, "$dir-$(ver.major).$(ver.minor)" for ver in cuda_versions["toolkit"])
        end
        append!(dirs, basedirs)
        push!(dirs, "/usr/lib/nvidia-cuda-toolkit")
    end

    # filter
    return valid_dirs(dirs)
end

# figure out the CUDA toolkit version (by looking at the `nvcc --version` output)
function find_toolkit_version(toolkit_dirs)
    nvcc_path = find_cuda_binary("nvcc", toolkit_dirs)
    if nvcc_path == nothing
        error("CUDA toolkit $(source_str(toolkit_dirs)) doesn't contain nvcc")
    end

    # parse the nvcc version string
    re = r"\bV(?<major>\d+).(?<minor>\d+).(?<patch>\d+)\b"
    m = match(re, read(`$nvcc_path --version`, String))
    m != nothing || error("Could not get version from nvcc")

    version = VersionNumber(parse(Int, m[:major]),
                            parse(Int, m[:minor]),
                            parse(Int, m[:patch]))
    @debug("CUDA toolkit identified as $version")
    return version
end

"""
    find_libdevice(targets::Vector{VersionNumber}, toolkit_dirs::Vector{String})

Look for the CUDA device library supporting `targets` in any of the CUDA toolkit directories
`toolkit_dirs`. On CUDA >= 9.0, a single library unified library is discovered and returned
as a string. On older toolkits, individual libraries for each of the targets are returned as
a vector of strings.
"""
function find_libdevice(targets::Vector{VersionNumber}, toolkit_dirs)
    @debug("Request to look $(source_str(toolkit_dirs)) for libdevice $(join(targets, ", "))")

    # figure out locations
    dirs = String[]
    for toolkit_dir in toolkit_dirs
        push!(dirs, joinpath(toolkit_dir, "libdevice"))
        push!(dirs, joinpath(toolkit_dir, "nvvm", "libdevice"))
    end

    # filter
    dirs = valid_dirs(dirs)
    if length(dirs) > 1
        warn("Found multiple libdevice locations: ", join(dirs, ", ", " and "))
    end
    @debug("Looking $(source_str(dirs)) for libdevice")

    # select
    if isempty(dirs)
        return nothing
    end
    dir = first(dirs)
    @debug("Found libdevice at $dir")

    # parse filenames
    libraries = Dict{VersionNumber,String}()
    for target in targets
        path = joinpath(dir, "libdevice.compute_$(target.major)$(target.minor).10.bc")
        if isfile(path)
            libraries[target] = path
        end
    end
    library = nothing
    let path = joinpath(dir, "libdevice.10.bc")
        if isfile(path)
            library = path
        end
    end

    # select
    if library != nothing
        @debug("Found unified libdevice at $library")
        return library
    elseif !isempty(libraries)
        @debug("Found libdevice for ", join(sort(map(ver->"sm_$(ver.major)$(ver.minor)",
                                            keys(libraries))), ", ", " and "), " at $dir")
        return libraries
    else
        error("No suitable libdevice found")
    end
end

function find_host_compiler(toolkit_version=nothing)
    if !(Compat.Sys.iswindows() || Compat.Sys.isapple())
        # Unix-like platforms: find compatible GCC binary

        # enumerate possible names for the gcc binary
        # NOTE: this is coarse, and might list invalid, non-existing versions
        gcc_names = [ "gcc" ]
        for major in 3:7
            push!(gcc_names, "gcc-$major")
            for minor in 0:9
                push!(gcc_names, "gcc-$major.$minor")
                push!(gcc_names, "gcc$major$minor")
            end
        end

        # find the binary
        gcc_possibilities = []
        for gcc_name in gcc_names
            # check if the binary exists
            gcc_path = find_binary([gcc_name])
            if gcc_path == nothing
                continue
            end

            # parse the GCC version string
            verstring = chomp(readlines(`$gcc_path --version`)[1])
            m = match(Regex("^$(basename(gcc_path)) \\(.*\\) ([0-9.]+)"), verstring)
            if m === nothing
                warn("Could not parse GCC version info (\"$verstring\"), skipping this compiler.")
                continue
            end
            gcc_ver = VersionNumber(m.captures[1])
            @debug("Found GCC $gcc_ver at $gcc_path")

            if toolkit_version == nothing || gcc_supported(gcc_ver, toolkit_version)
                push!(gcc_possibilities, (gcc_path, gcc_ver))
            end
        end

        # select the most recent compiler
        if length(gcc_possibilities) == 0
            error("Could not find a suitable GCC")
        end
        sort!(gcc_possibilities; rev=true, lt=(a, b) -> a[2]<b[2])
        host_compiler, host_version = gcc_possibilities[1]
    elseif Compat.Sys.iswindows()
        # Windows: search for MSVC in default locations

        # discover Visual Studio installations
        msvc_paths = String[]
        arch = Sys.WORD_SIZE == 64 ? "amd64" : "x86"
        program_files = ENV[Sys.WORD_SIZE == 64 ? "ProgramFiles(x86)" : "ProgramFiles"]
        ## locate VS2017
        vswhere_dist = joinpath(program_files, "Microsoft Visual Studio", "Installer", "vswhere.exe")
        let vswhere = isfile(vswhere_dist) ? vswhere_dist : download("https://github.com/Microsoft/vswhere/releases/download/2.2.11/vswhere.exe")
            msvc_cmd_tools_dir = chomp(read(`$vswhere -latest -property installationPath`, String))
            if !isempty(msvc_cmd_tools_dir)
                vs_prompt = joinpath(msvc_cmd_tools_dir, "VC", "Auxiliary", "Build", "vcvarsall.bat")
                tmpfile = tempname() # TODO: do this with a pipe
                run(pipeline(`$vs_prompt $arch \& where cl.exe`, tmpfile))
                msvc_path = readlines(tmpfile)[end]
                @debug("Considering MSVC at $msvc_path located with vswhere")
                push!(msvc_paths, msvc_path)
            end
        end
        ## locate VS2012 to 2014
        let envvars = ["VS140COMNTOOLS", "VS120COMNTOOLS", "VS110COMNTOOLS", "VS100COMNTOOLS"]
            envvars_set = filter(var -> haskey(ENV, var), envvars)
            for var in envvars_set
                val = ENV[var]
                msvc_path = joinpath(dirname(dirname(dirname(val))), "VC", "bin", arch, "cl.exe")
                if isfile(msvc_path)
                    @debug("Considering MSVC at $msvc_path located with environment variable $var")
                    push!(msvc_paths, msvc_path)
                end
            end
        end
        isempty(msvc_paths) && error("No Visual Studio installation found")

        # find MSVC versions
        msvc_list = Dict{VersionNumber,String}()
        for path in msvc_paths
            tmpfile = tempname() # TODO: do this with a pipe
            if !success(pipeline(`$path`, stdout=DevNull, stderr=tmpfile))
                warn("Could not execute $path")
                continue
            end
            ver_str = match(r"Version\s+(\d+(\.\d+)?(\.\d+)?)"i, read(tmpfile, String))[1]
            ver = VersionNumber(ver_str)
            msvc_list[ver] = path
        end

        # check compiler compatibility
        msvc_path, msvc_ver = nothing, nothing
        if toolkit_version != nothing
            for ver in sort(collect(keys(msvc_list)), rev=true) # search the highest version first
                if msvc_supported(ver, toolkit_version)
                    msvc_path, msvc_ver = msvc_list[ver], ver
                    break
                end
            end
            if msvc_ver == nothing
                error("None of the available Visual Studio C++ compilers ($(join(keys(msvc_list), ", "))) are compatible with CUDA $toolkit_version")
            end
        else
            # take the first found host, which will be the highest version found
            host_pair = first(sort(collect(msvc_list), by=x->x[1], rev=true))
            msvc_path, msvc_ver = last(host_pair), first(host_pair)
        end

        host_compiler, host_version = msvc_path, msvc_ver
    elseif Compat.Sys.isapple()
        # GCC is no longer supported on MacOS so let's just use clang
        # TODO: discovery of all compilers, and version matching against the toolkit
        clang_path = find_binary(["clang"])
        if clang_path == nothing
            error("Could not find clang")
        end
        clang_ver_str = match(r"version\s+(\d+(\.\d+)?(\.\d+)?)"i, readstring(`$clang_path --version`))[1]
        clang_ver = VersionNumber(clang_ver_str)

        host_compiler, host_version = clang_path, clang_ver
    end
    @debug("Selected host compiler version $host_version at $host_compiler")

    return host_compiler, host_version
end

mutable struct Toolchain
    cuda_compiler::String
    cuda_version::VersionNumber

    host_compiler::String
    host_version::VersionNumber
end
function find_toolchain(toolkit_dirs, toolkit_version=find_toolkit_version(toolkit_dirs))
    # find the CUDA compiler
    nvcc_path = find_cuda_binary("nvcc", toolkit_dirs)
    if nvcc_path == nothing
        error("CUDA toolkit $(source_str(toolkit_dirs)) doesn't contain nvcc")
    end
    nvcc_version = toolkit_version

    # find a suitable host compiler
    host_compiler, host_version = find_host_compiler(toolkit_version)

    return Toolchain(nvcc_path, nvcc_version,
                     host_compiler, host_version)
end
