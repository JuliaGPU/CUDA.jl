export find_cuda_library, find_cuda_binary,
       find_toolkit, parse_toolkit_version,
       find_libdevice, find_libcudadevrt

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
                      word_size::Integer=Sys.WORD_SIZE)
    @trace "Request to look for library $(join(names, ", "))" locations

    # figure out names
    all_names = String[]
    if Sys.iswindows()
        # priority goes to the `names` argument, as per `Libdl.find_library`
        for name in names
            # first look for unversioned libraries, for upgrade resilience
            append!(all_names, ["$(name)$(word_size)", name])
            for version in versions
                append!(all_names, ["$(name)$(word_size)_$(version.major)",
                                    "$(name)$(word_size)_$(version.major)$(version.minor)"])
            end
        end
    elseif Sys.isunix()
        # most UNIX distributions ship versioned libraries (also see JuliaLang/julia#22828)
        for name in names
            # first look for unversioned libraries, for upgrade resilience
            push!(all_names, "lib$(name).$(Libdl.dlext)")
            for version in versions
                append!(all_names, ["lib$(name).$(Libdl.dlext).$(version.major)",
                                    "lib$(name).$(Libdl.dlext).$(version.major).$(version.minor)"])
            end
        end
    else
        # let Libdl do all the work
        all_names = ["lib$name" for name in names]
    end
    unique!(all_names)

    # figure out locations
    all_locations = String[]
    for location in locations
        push!(all_locations, location)
        if Sys.iswindows()
            push!(all_locations, joinpath(location, "bin"))
        else
            push!(all_locations, joinpath(location, "lib"))
            if word_size == 64
                push!(all_locations, joinpath(location, "lib64"))
            end
        end
    end

    @trace "Looking for library $(join(all_names, ", "))" locations=all_locations
    name_found = Libdl.find_library(all_names, all_locations)
    if isempty(name_found)
        return nothing
    end

    # find the full path of the library (which Libdl.find_library doesn't guarantee to return)
    path = Libdl.dlpath(name_found)
    @debug "Found library $(basename(path)) at $(dirname(path))"
    return path
end

"""
    find_binary(names; locations=String[])

Similar to `find_library`, performs an exhaustive search for a binary in various
subdirectories of `locations`, and finally PATH.
"""
function find_binary(names::Vector{String};
                     locations::Vector{String}=String[])
    @trace "Request to look for binary $(join(names, ", "))" locations

    # figure out names
    all_names = String[]
    if Sys.iswindows()
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
        dirs = split(path, Sys.iswindows() ? ';' : ':')
        filter!(path->!isempty(path), dirs)
        append!(all_locations, dirs)
    end

    @trace "Looking for binary $(join(all_names, ", "))" locations=all_locations
    all_paths = [joinpath(location, name) for name in all_names, location in all_locations]
    paths = String[]
    for path in all_paths
        try
            if ispath(path)
                push!(paths, path)
            end
        catch
            # some system disallow `stat` on certain paths
        end
    end

    if isempty(paths)
        return nothing
    else
        path = first(paths)
        @debug "Found binary $(basename(path)) at $(dirname(path))"
        return path
    end
end


## CUDA-specific discovery routines

const cuda_names = Dict(
    "cuda"      => Sys.iswindows() ? ["nvcuda"] : ["cuda"],
    "nvml"      => Sys.iswindows() ? ["nvml"]   : ["nvidia-ml"],
    "nvtx"      => ["nvToolsExt"]
)

const cuda_versions = [v"1.0", v"1.1",
                       v"2.0", v"2.1", v"2.2",
                       v"3.0", v"3.1", v"3.2",
                       v"4.0", v"4.1", v"4.2",
                       v"5.0", v"5.5",
                       v"6.0", v"6.5",
                       v"7.0", v"7.5",
                       v"8.0",
                       v"9.0", v"9.1", v"9.2",
                       v"10.0", v"10.1", v"10.2"]

# simplified find_library/find_binary entry-points,
# looking up name aliases and known version numbers
# and passing the (optional) toolkit dirs as locations.
find_cuda_library(name::String, toolkit_dirs::Vector{String}=String[],
                  versions::Vector{VersionNumber}=VersionNumber[]; kwargs...) =
    find_library(get(cuda_names, name, [name]);
                 versions=versions, locations=toolkit_dirs, kwargs...)
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
    dirs = String[]

    # NVTX library (special case for Windows)
    if Sys.iswindows()
        var = "NVTOOLSEXT_PATH"
        basedir = get(ENV, var, nothing)
        if basedir !== nothing && isdir(basedir)
            @trace "Looking for NVTX library via environment variable" var
            suffix = Sys.WORD_SIZE == 64 ? "x64" : "Win32"
            dir = joinpath(basedir, "bin", suffix)
            isdir(dir) && push!(dirs, dir)
        else
            program_files = ENV[Sys.WORD_SIZE == 64 ? "ProgramFiles" : "ProgramFiles(x86)"]
            basedir = joinpath(program_files, "NVIDIA Corporation", "NvToolsExt")
            @trace "Looking for NVTX library in the default directory" basedir
            suffix = Sys.WORD_SIZE == 64 ? "x64" : "Win32"
            dir = joinpath(basedir, "bin", suffix)
            isdir(dir) && push!(dirs, dir)
        end
    end

    # look for environment variables to override discovery
    envvars = ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"]
    filter!(var -> haskey(ENV, var) && ispath(ENV[var]), envvars)
    if !isempty(envvars)
        paths = unique(map(var->ENV[var], envvars))
        if length(paths) > 1
            @warn "Multiple CUDA environment variables set to different values: $(join(paths, ", "))"
        end

        @trace "Looking for CUDA toolkit via environment variables $(join(envvars, ", "))"
        append!(dirs, paths)
        return dirs
    end


    # look for the compiler binary (in the case PATH points to the installation)
    ptxas_path = find_cuda_binary("ptxas")
    if ptxas_path !== nothing
        ptxas_dir = dirname(ptxas_path)
        if occursin(r"^bin(32|64)?$", basename(ptxas_dir))
            ptxas_dir = dirname(ptxas_dir)
        end

        @trace "Looking for CUDA toolkit via ptxas binary" path=ptxas_path dir=ptxas_dir
        push!(dirs, ptxas_dir)
    end

    # look for the runtime library (in the case LD_LIBRARY_PATH points to the installation)
    libcudart_path = find_cuda_library("cudart", String[], cuda_versions)
    if libcudart_path !== nothing
        libcudart_dir = dirname(libcudart_path)
        if occursin(r"^(lib|bin)(32|64)?$", basename(libcudart_dir))
            libcudart_dir = dirname(libcudart_dir)
        end

        @trace "Looking for CUDA toolkit via CUDA runtime library" path=libcudart_path dir=libcudart_dir
        push!(dirs, libcudart_dir)
    end

    # look in default installation directories
    default_dirs = String[]
    if Sys.iswindows()
        # CUDA versions are installed in separate directories under a single base dir
        program_files = ENV[Sys.WORD_SIZE == 64 ? "ProgramFiles" : "ProgramFiles(x86)"]
        basedir = joinpath(program_files, "NVIDIA GPU Computing Toolkit", "CUDA")
        if isdir(basedir)
            entries = map(dir -> joinpath(basedir, dir), readdir(basedir))
            append!(default_dirs, entries)
        end
    else
        # CUDA versions are installed in unversioned dirs, or suffixed with the version
        basedirs = ["/usr/local/cuda", "/opt/cuda"]
        for ver in cuda_versions, dir in basedirs
            push!(default_dirs, "$dir-$(ver.major).$(ver.minor)")
        end
        append!(default_dirs, basedirs)
        push!(default_dirs, "/usr/lib/nvidia-cuda-toolkit")
        push!(default_dirs, "/usr/share/cuda")
    end
    reverse!(default_dirs) # we want to search starting from the newest CUDA version
    default_dirs = valid_dirs(default_dirs)
    if !isempty(default_dirs)
        @trace "Looking for CUDA toolkit via default installation directories" dirs=default_dirs
        append!(dirs, default_dirs)
    end

    # filter
    dirs = valid_dirs(dirs)
    @debug "Found CUDA toolkit at $(join(dirs, ", "))"
    return dirs
end

# figure out the CUDA toolkit version (by looking at the output of a tool like `nvdisasm`)
function parse_toolkit_version(tool_path)
    # parse the version string
    verstr = withenv("LANG"=>"C") do
        read(`$tool_path --version`, String)
    end
    m = match(r"\bV(?<major>\d+).(?<minor>\d+).(?<patch>\d+)\b", verstr)
    m !== nothing || error("could not parse version info (\"$verstr\")")

    version = VersionNumber(parse(Int, m[:major]),
                            parse(Int, m[:minor]),
                            parse(Int, m[:patch]))
    @debug "CUDA toolkit identified as $version"
    return version
end

"""
    find_libdevice(toolkit_dirs::Vector{String})

Look for the CUDA device library supporting `targets` in any of the CUDA toolkit directories
`toolkit_dirs`. On CUDA >= 9.0, a single library unified library is discovered and returned
as a string. On older toolkits, individual libraries for each of the targets are returned as
a vector of strings.
"""
function find_libdevice(toolkit_dirs)
    @trace "Request to look for libdevice" locations=toolkit_dirs

    # figure out locations
    dirs = String[]
    for toolkit_dir in toolkit_dirs
        push!(dirs, toolkit_dir)
        push!(dirs, joinpath(toolkit_dir, "libdevice"))
        push!(dirs, joinpath(toolkit_dir, "nvvm", "libdevice"))
    end

    # filter
    dirs = valid_dirs(dirs)
    @trace "Look for libdevice" locations=dirs

    for dir in dirs
        path = joinpath(dir, "libdevice.10.bc")
        if isfile(path)
            @debug "Found unified device library at $path"
            return path
        end
    end

    return nothing
end

"""
    find_libcudadevrt(toolkit_dirs::Vector{String})

Look for the CUDA device runtime library in any of the CUDA toolkit directories
`toolkit_dirs`.
"""
function find_libcudadevrt(toolkit_dirs)
    locations = toolkit_dirs
    @trace "Request to look for libcudadevrt " locations

    name = nothing
    if Sys.isunix()
        name = "libcudadevrt.a"
    elseif Sys.iswindows()
        name = "cudadevrt.lib"
    else
        error("No support for discovering the CUDA device runtime library on your platform, please file an issue.")
    end

    # figure out locations
    all_locations = String[]
    for location in locations
        push!(all_locations, location)
        if Sys.iswindows()
            if Sys.WORD_SIZE == 64
                push!(all_locations, joinpath(location, "lib", "x64"))
            elseif Sys.WORD_SIZE == 32
                push!(all_locations, joinpath(location, "lib", "Win32"))
            end
        else
            push!(all_locations, joinpath(location, "lib"))
            if Sys.WORD_SIZE == 64
                push!(all_locations, joinpath(location, "lib64"))
            end
        end
    end

    @trace "Looking for CUDA device runtime library $name" locations=all_locations
    paths = filter(isfile, map(location->joinpath(location, name), all_locations))

    if isempty(paths)
        return nothing
    else
        path = first(paths)
        @debug "Found CUDA device runtime library $(basename(path)) at $(dirname(path))"
        return path
    end
end
