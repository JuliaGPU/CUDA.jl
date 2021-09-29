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

function join_versions(versions)
    isempty(versions) && return "no specific version"
    "version " * join(versions, " or ")
end

function join_locations(locations)
    isempty(locations) && return "in no specific location"
    "in " * join(locations, " or ")
end


## generic discovery routines

function library_names(name::String, versions::Vector=[])
    names = String[]

    # always look for an unversioned library first
    if Sys.iswindows()
        push!(names, "$(name)$(Sys.WORD_SIZE).$(Libdl.dlext)")

        # some libraries (e.g. CUTENSOR) are shipped without the word size-prefix
        push!(names, "$(name).$(Libdl.dlext)")
    elseif Sys.isapple()
        # macOS puts the version number before the dylib extension
        push!(names, "lib$(name).$(Libdl.dlext)")
    elseif Sys.isunix()
        # most UNIX distributions ship versioned libraries (also see JuliaLang/julia#22828)
        push!(names, "lib$(name).$(Libdl.dlext)")
    else
        push!(names, "lib$name.$(Libdl.dlext)")
    end

    # then consider versioned libraries
    for version in versions
        if Sys.iswindows()
            # Windows encodes the version in the filename
            if version isa VersionNumber
                append!(names, ["$(name)$(Sys.WORD_SIZE)_$(version.major)$(version.minor).$(Libdl.dlext)",
                                "$(name)$(Sys.WORD_SIZE)_$(version.major).$(Libdl.dlext)"])
            elseif version isa String
                push!(names, "$(name)$(Sys.WORD_SIZE)_$(version).$(Libdl.dlext)")
            end

            # some libraries (e.g. CUTENSOR) are shipped without the word size-prefix
            if version isa VersionNumber
                append!(names, ["$(name)_$(version.major)$(version.minor).$(Libdl.dlext)",
                                "$(name)_$(version.major).$(Libdl.dlext)"])
            elseif version isa String
                push!(names, "$(name)_$(version).$(Libdl.dlext)")
            end
        elseif Sys.isapple()
            # macOS puts the version number before the dylib extension
            if version isa VersionNumber
                append!(names, ["lib$(name).$(version.major).$(version.minor).$(Libdl.dlext)",
                                "lib$(name).$(version.major).$(Libdl.dlext)"])
            elseif version isa String
                push!(names, "lib$(name).$(version).$(Libdl.dlext)")
            end
        elseif Sys.isunix()
            # most UNIX distributions ship versioned libraries (also see JuliaLang/julia#22828)
            if version isa VersionNumber
                append!(names, ["lib$(name).$(Libdl.dlext).$(version.major).$(version.minor).$(version.patch)",
                                "lib$(name).$(Libdl.dlext).$(version.major).$(version.minor)",
                                "lib$(name).$(Libdl.dlext).$(version.major)"])
            elseif version isa String
                push!(names, "lib$(name).$(Libdl.dlext).$(version)")
            end
        end
    end

    return names
end

"""
    find_library(name, versions; locations=String[])

Wrapper for Libdl.find_library, performing a more exhaustive search:

- variants of the library name (including version numbers, platform-specific tags, etc);
- various subdirectories of the `locations` list, and finally system library directories.

Returns the full path to the library.
"""
function find_library(name::String, versions::Vector=[];
                      locations::Vector{String}=String[])
    # figure out names
    all_names = library_names(name, versions)

    # figure out locations
    all_locations = String[]
    for location in locations
        push!(all_locations, location)
        push!(all_locations, joinpath(location, "lib"))
        if Sys.WORD_SIZE == 64
            push!(all_locations, joinpath(location, "lib64"))
            push!(all_locations, joinpath(location, "libx64"))
        end
        if Sys.iswindows()
            push!(all_locations, joinpath(location, "bin"))
            push!(all_locations, joinpath(location, "bin", Sys.WORD_SIZE==64 ? "x64" : "Win32"))
        end
    end

    @debug "Looking for library $name, $(join_versions(versions)), $(join_locations(locations))" all_names all_locations
    name_found = Libdl.find_library(all_names, all_locations)
    if isempty(name_found)
        @debug "Did not find $name"
        return nothing
    end

    # find the full path of the library (which Libdl.find_library doesn't guarantee to return)
    path = Libdl.dlpath(name_found)
    @debug "Found $(basename(path)) at $(dirname(path))"
    return path
end

"""
    find_binary(name; locations=String[])

Similar to `find_library`, performs an exhaustive search for a binary in various
subdirectories of `locations`, and finally PATH by using `Sys.which`.
"""
function find_binary(name::String; locations::Vector{String}=String[])
    # figure out locations
    all_locations = String[]
    for location in locations
        push!(all_locations, location)
        push!(all_locations, joinpath(location, "bin"))
    end
    # we look in PATH too by using `Sys.which` with unadorned names

    @debug "Looking for binary $name $(join_locations(locations))" all_locations
    all_paths = [name; [joinpath(location, name) for location in all_locations]]
    for path in all_paths
        try
            program_path = Sys.which(path)
            if program_path !== nothing
                @debug "Found $path at $program_path"
                return program_path
            end
        catch
            # some system disallow `stat` on certain paths
        end
    end

    @debug "Did not find $path"
    return nothing
end


## CUDA-specific discovery routines

const cuda_releases = [v"9.0", v"9.1", v"9.2",
                       v"10.0", v"10.1", v"10.2",
                       v"11.0", v"11.1", v"11.2", v"11.3", v"11.4"]

# return possible versions of a CUDA library
function cuda_library_versions(name::String)
    if Sys.iswindows()
        # CUDA libraries on Windows are always versioned, however, we don't
        # know which version we're looking for (and we don't first want to
        # figure that out by, say, invoking a versionless binary like ptxas).

        # start out with all known CUDA releases
        versions = Any[cuda_releases...]

        # append some future releases
        for major in last(versions).major:15, minor in 1:10
            version = VersionNumber(major, minor)
            if !in(version, versions)
                push!(versions, version)
            end
        end

        # CUPTI is special, and uses a dot-separated, year-based versioning
        if name == "cupti"
            for year in 2020:2022, major in 1:5, minor in 0:3
                version = "$year.$major.$minor"
                push!(versions, version)
            end
        end

        # NVTX is special, and only uses a single digit
        if name == "nvToolsExt"
            append!(versions, [v"1", v"2"])
        end

        versions
    else
        # only consider unversioned libraries on other platforms.
        []
    end
end

# simplified find_library/find_binary entry-points,
# looking up name aliases and known version numbers
# and passing the (optional) toolkit dirs as locations.
function find_cuda_library(toolkit_dirs::Vector{String}, library::String, versions::Vector)
    # figure out the location
    locations = toolkit_dirs
    ## CUPTI is in the "extras" directory of the toolkit
    if library == "cupti"
        toolkit_extras_dirs = filter(dir->isdir(joinpath(dir, "extras")), toolkit_dirs)
        cupti_dirs = map(dir->joinpath(dir, "extras", "CUPTI"), toolkit_extras_dirs)
        append!(locations, cupti_dirs)
    end
    ## NVTX is located in an entirely different location on Windows
    if library == "nvToolsExt" && Sys.iswindows()
        if haskey(ENV, "NVTOOLSEXT_PATH")
            dir = ENV["NVTOOLSEXT_PATH"]
            @debug "Looking for NVTX library via environment variable" dir
        else
            program_files = ENV[Sys.WORD_SIZE == 64 ? "ProgramFiles" : "ProgramFiles(x86)"]
            dir = joinpath(program_files, "NVIDIA Corporation", "NvToolsExt")
            @debug "Looking for NVTX library in the default directory" dir
        end
        isdir(dir) && push!(locations, dir)
    end

    find_library(library, versions; locations)
end
function find_cuda_binary(toolkit_dirs::Vector{String}, name::String)
    # figure out the location
    locations = toolkit_dirs
    ## compute-sanitizer is in the "extras" directory of the toolkit
    if name == "compute-sanitizer"
        toolkit_extras_dirs = filter(dir->isdir(joinpath(dir, "extras")), toolkit_dirs)
        sanitizer_dirs = map(dir->joinpath(dir, "extras", "compute-sanitizer"), toolkit_extras_dirs)
        append!(locations, sanitizer_dirs)
    end

    find_binary(name; locations)
end

"""
    find_toolkit()::Vector{String}

Look for directories where (parts of) the CUDA toolkit might be installed. This returns a
(possibly empty) list of paths that can be used as an argument to other discovery functions.

The behavior of this function can be overridden by defining the `CUDA_PATH`, `CUDA_HOME` or
`CUDA_ROOT` environment variables, which should point to the root of the CUDA toolkit.
"""
function find_toolkit()
    dirs = String[]

    # look for environment variables to override discovery
    envvars = ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"]
    filter!(var -> haskey(ENV, var) && ispath(ENV[var]), envvars)
    if !isempty(envvars)
        paths = unique(map(var->ENV[var], envvars))
        if length(paths) > 1
            @warn "Multiple CUDA environment variables set to different values: $(join(paths, ", "))"
        end

        @debug "Looking for CUDA toolkit via environment variables $(join(envvars, ", "))"
        append!(dirs, paths)
        return dirs
    end

    # look for the compiler binary (in the case PATH points to the installation)
    ptxas_path = find_binary("ptxas")
    if ptxas_path !== nothing
        ptxas_dir = dirname(ptxas_path)
        if occursin(r"^bin(32|64)?$", basename(ptxas_dir))
            ptxas_dir = dirname(ptxas_dir)
        end

        @debug "Looking for CUDA toolkit via ptxas binary" path=ptxas_path dir=ptxas_dir
        push!(dirs, ptxas_dir)
    end

    # look for the runtime library (in the case LD_LIBRARY_PATH points to the installation)
    libcudart_path = find_library("cudart")
    if libcudart_path !== nothing
        libcudart_dir = dirname(libcudart_path)
        if occursin(r"^(lib|bin)(32|64)?$", basename(libcudart_dir))
            libcudart_dir = dirname(libcudart_dir)
        end

        @debug "Looking for CUDA toolkit via CUDA runtime library" path=libcudart_path dir=libcudart_dir
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
        for ver in cuda_releases, dir in basedirs
            push!(default_dirs, "$dir-$(ver.major).$(ver.minor)")
        end
        append!(default_dirs, basedirs)
        push!(default_dirs, "/usr/lib/nvidia-cuda-toolkit")
        push!(default_dirs, "/usr/share/cuda")
    end
    reverse!(default_dirs) # we want to search starting from the newest CUDA version
    default_dirs = valid_dirs(default_dirs)
    if !isempty(default_dirs)
        @debug "Looking for CUDA toolkit via default installation directories" dirs=default_dirs
        append!(dirs, default_dirs)
    end

    # filter
    dirs = valid_dirs(dirs)
    @debug "Found CUDA toolkit at $(join(dirs, ", "))"
    return dirs
end

"""
    find_libdevice(toolkit_dirs::Vector{String})

Look for the CUDA device library supporting `targets` in any of the CUDA toolkit directories
`toolkit_dirs`. On CUDA >= 9.0, a single library unified library is discovered and returned
as a string. On older toolkits, individual libraries for each of the targets are returned as
a vector of strings.
"""
function find_libdevice(toolkit_dirs)
    @debug "Request to look for libdevice" locations=toolkit_dirs

    # figure out locations
    dirs = String[]
    for toolkit_dir in toolkit_dirs
        push!(dirs, toolkit_dir)
        push!(dirs, joinpath(toolkit_dir, "libdevice"))
        push!(dirs, joinpath(toolkit_dir, "nvvm", "libdevice"))
    end

    # filter
    dirs = valid_dirs(dirs)
    @debug "Look for libdevice" locations=dirs

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
    @debug "Request to look for libcudadevrt " locations

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

    @debug "Looking for CUDA device runtime library $name" locations=all_locations
    paths = filter(isfile, map(location->joinpath(location, name), all_locations))

    if isempty(paths)
        return nothing
    else
        path = first(paths)
        @debug "Found CUDA device runtime library $(basename(path)) at $(dirname(path))"
        return path
    end
end
