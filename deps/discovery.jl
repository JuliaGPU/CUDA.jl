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

function library_versioned_names(name::String, version::Union{Nothing,VersionNumber,String}=nothing)
    names = String[]
    if Sys.iswindows()
        # Windows encodes the version in the filename
        if version isa VersionNumber
            append!(names, ["$(name)$(Sys.WORD_SIZE)_$(version.major)$(version.minor).$(Libdl.dlext)",
                            "$(name)$(Sys.WORD_SIZE)_$(version.major).$(Libdl.dlext)"])
        elseif version isa String
            push!(names, "$(name)$(Sys.WORD_SIZE)_$(version).$(Libdl.dlext)")
        elseif version === nothing
            push!(names, "$(name)$(Sys.WORD_SIZE).$(Libdl.dlext)")
        end

        # some libraries (e.g. CUTENSOR) are shipped without the word size-prefix
        if version isa VersionNumber
            append!(names, ["$(name)_$(version.major)$(version.minor).$(Libdl.dlext)",
                            "$(name)_$(version.major).$(Libdl.dlext)"])
        elseif version isa String
            push!(names, "$(name)_$(version).$(Libdl.dlext)")
        elseif version === nothing
            push!(names, "$(name).$(Libdl.dlext)")
        end
    elseif Sys.isapple()
        # macOS puts the version number before the dylib extension
        if version isa VersionNumber
            append!(names, ["lib$(name).$(version.major).$(version.minor).$(Libdl.dlext)",
                            "lib$(name).$(version.major).$(Libdl.dlext)"])
        elseif version isa String
            push!(names, "lib$(name).$(version).$(Libdl.dlext)")
        elseif version === nothing
            push!(names, "lib$(name).$(Libdl.dlext)")
        end
    elseif Sys.isunix()
        # most UNIX distributions ship versioned libraries (also see JuliaLang/julia#22828)
        if version isa VersionNumber
            append!(names, ["lib$(name).$(Libdl.dlext).$(version.major).$(version.minor)",
                            "lib$(name).$(Libdl.dlext).$(version.major)"])
        elseif version isa String
            push!(names, "lib$(name).$(Libdl.dlext).$(version)")
        elseif version === nothing
            push!(names, "lib$(name).$(Libdl.dlext)")
        end
    elseif version === nothing
        push!(names, "lib$name.$(Libdl.dlext)")
    end
    return names
end

"""
    find_library(name, version; locations=String[])

Wrapper for Libdl.find_library, performing a more exhaustive search:

- variants of the library name (including version numbers, platform-specific tags, etc);
- various subdirectories of the `locations` list, and finally system library directories.

Returns the full path to the library.
"""
function find_library(name::String, version::Union{Nothing,VersionNumber,String}=nothing;
                      locations::Vector{String}=String[])
    @debug "Request to look for library $name $version" locations

    # figure out names
    all_names = library_versioned_names(name, version)

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

    @debug "Looking for library $(join(all_names, ", "))" locations=all_locations
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
    find_binary(name; locations=String[])

Similar to `find_library`, performs an exhaustive search for a binary in various
subdirectories of `locations`, and finally PATH by using `Sys.which`.
"""
function find_binary(name::String; locations::Vector{String}=String[])
    @debug "Request to look for binary $name" locations

    # figure out locations
    all_locations = String[]
    for location in locations
        push!(all_locations, location)
        push!(all_locations, joinpath(location, "bin"))
    end
    # we look in PATH too by using `Sys.which` with unadorned names

    @debug "Looking for binary $name" locations=all_locations
    all_paths = [name; [joinpath(location, name) for location in all_locations]]
    for path in all_paths
        try
            program_path = Sys.which(path)
            if program_path !== nothing
                @debug "Found binary $path at $program_path"
                return program_path
            end
        catch
            # some system disallow `stat` on certain paths
        end
    end

    return nothing
end


## CUDA-specific discovery routines

const cuda_releases = [v"1.0", v"1.1",
                       v"2.0", v"2.1", v"2.2",
                       v"3.0", v"3.1", v"3.2",
                       v"4.0", v"4.1", v"4.2",
                       v"5.0", v"5.5",
                       v"6.0", v"6.5",
                       v"7.0", v"7.5",
                       v"8.0",
                       v"9.0", v"9.1", v"9.2",
                       v"10.0", v"10.1", v"10.2",
                       v"11.0", v"11.1"]

const cuda_library_versions = Dict(
    v"11.0.2" => Dict(
        "cudart"    => v"11.0.171",
        "cupti"     => "2020.1.0", # wtf
        "nvrtc"     => v"11.0.167",
        "nvtx"      => v"11.0.167",
        "nvvp"      => v"11.0.167",
        "cublas"    => v"11.0.0", #.191
        "cufft"     => v"10.1.3", #.191
        "curand"    => v"10.2.0", #.191
        "cusolver"  => v"10.4.0", #.191
        "cusparse"  => v"11.0.0", #.191
        "npp"       => v"11.0.0", #.191
        "nvjpeg"    => v"11.0.0", #.191
    ),
    v"11.0.3" => Dict(
        "cudart"    => v"11.0.221",
        "cupti"     => "2020.1.1", # docs mention 11.0.221
        "nvrtc"     => v"11.0.221",
        "nvtx"      => v"11.0.167",
        "nvvp"      => v"11.0.221",
        "cublas"    => v"11.2.0", #.252
        "cufft"     => v"10.2.1", #.245
        "curand"    => v"10.2.1", #.245
        "cusolver"  => v"10.6.0", #.245
        "cusparse"  => v"11.1.1", #.245
        "npp"       => v"11.1.0", #.245
        "nvjpeg"    => v"11.1.1", #.245
    ),
    v"11.1.0" => Dict(
        "cudart"    => v"11.1.74",
        "cupti"     => "2020.2.0", # docs mention 11.1.69
        "nvrtc"     => v"11.1.74",
        "nvtx"      => v"11.1.74",
        "nvvp"      => v"11.1.74",
        "cublas"    => v"11.2.1", #.74
        "cufft"     => v"10.3.0", #.74
        "curand"    => v"10.2.2", #.74
        "cusolver"  => v"11.0.0", #.74
        "cusparse"  => v"11.2.0", #.275
        "npp"       => v"11.1.1", #.269
        "nvjpeg"    => v"11.2.0", #.74
    ),
    v"11.1.1" => Dict(
        "cudart"    => v"11.1.74",
        "cupti"     => "2020.2.1", # docs mention 11.1.105
        "nvrtc"     => v"11.1.105",
        "nvtx"      => v"11.1.74",
        "nvvp"      => v"11.1.105",
        "cublas"    => v"11.3.0", #.106
        "cufft"     => v"10.3.0", #.105
        "curand"    => v"10.2.2", #.105
        "cusolver"  => v"11.0.1", #.105
        "cusparse"  => v"11.3.0", #.10
        "npp"       => v"11.1.2", #.105
        "nvjpeg"    => v"11.3.0", #.105
    ),
)

function cuda_library_version(library, toolkit_version)
    if library == "nvtx"
        v"1"
    elseif toolkit_version >= v"11"
        # starting with CUDA 11, libraries are versioned independently
        if !haskey(cuda_library_versions, toolkit_version)
            error("CUDA.jl does not yet support CUDA $toolkit_version; please file an issue.")
        end
        cuda_library_versions[toolkit_version][library]
    else
        toolkit_version
    end
end

const cuda_library_names = Dict(
    "nvtx"      => "nvToolsExt"
)

# only for nvdisasm, to discover the CUDA toolkit version
const cuda_binary_versions = Dict(
    v"11.0.2" => Dict(
        "nvdisasm"  => v"11.0.194"
    ),
    v"11.0.3" => Dict(
        "nvdisasm"  => v"11.0.221"
    ),
    v"11.1.0" => Dict(
        "nvdisasm"  => v"11.1.74"
    ),
    v"11.1.1" => Dict(
        "nvdisasm"  => v"11.1.74"   # ambiguous!
    ),
)

# simplified find_library/find_binary entry-points,
# looking up name aliases and known version numbers
# and passing the (optional) toolkit dirs as locations.
function find_cuda_library(library::String, toolkit_dirs::Vector{String},
                           toolkit_version::VersionNumber)
    toolkit_release = VersionNumber(toolkit_version.major, toolkit_version.minor)

    # figure out the location
    locations = toolkit_dirs
    ## CUPTI is in the "extras" directory of the toolkit
    if library == "cupti"
        toolkit_extras_dirs = filter(dir->isdir(joinpath(dir, "extras")), toolkit_dirs)
        cupti_dirs = map(dir->joinpath(dir, "extras", "CUPTI"), toolkit_extras_dirs)
        append!(locations, cupti_dirs)
    end
    ## NVTX is located in an entirely different location on Windows
    if library == "nvtx" && Sys.iswindows()
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

    version = cuda_library_version(library, toolkit_version)
    name = get(cuda_library_names, library, library)
    find_library(name, version; locations=locations)
end
find_cuda_binary(name::String, toolkit_dirs::Vector{String}=String[]) =
    find_binary(name; locations=toolkit_dirs)

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

# figure out the CUDA toolkit version (by looking at the output of a tool like `nvdisasm`)
function parse_toolkit_version(tool, tool_path::String)
    # parse the version string
    verstr = withenv("LANG"=>"C") do
        read(`$tool_path --version`, String)
    end
    m = match(r"\bV(?<major>\d+).(?<minor>\d+).(?<patch>\d+)\b", verstr)
    m !== nothing || error("could not parse version info (\"$verstr\")")

    version = VersionNumber(parse(Int, m[:major]),
                            parse(Int, m[:minor]),
                            parse(Int, m[:patch]))

    if version >= v"11"
        # starting with CUDA 11, binaries are versioned independently
        # NOTE: we can't always tell, e.g. nvdisasm is the same in CUDA 11.1.0 and 11.1.1.
        #       return the lowest version to ensure compatibility.
        for toolkit_version in sort(collect(keys(cuda_binary_versions)))
            if cuda_binary_versions[toolkit_version][tool] == version
                @debug "CUDA toolkit identified as $toolkit_version (providing $tool $version)"
                return toolkit_version
            end
        end
        error("CUDA.jl does not yet support CUDA with $tool $version; please file an issue.")
    else
        @debug "CUDA toolkit identified as $version"
        return version
    end
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
