export find_library, find_binary,
       find_cuda_library, find_cuda_binary,
       find_toolkit, find_toolkit_version,
       find_libdevice, find_libcudadevrt,
       find_host_compiler, find_toolchain

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
            if Sys.WORD_SIZE == 64
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
    @debug "Found library $name_found at $path"
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

# FIXME: CUDA on 32-bit Windows isn't supported

const cuda_names = Dict(
    "cuda"      => Sys.iswindows() ? ["nvcuda"] : ["cuda"],
    "nvml"      => Sys.iswindows() ? ["nvml"]   : ["nvidia-ml"]
)

const cuda_versions = Dict(
    # https://developer.nvidia.com/cuda-toolkit-archive
    "toolkit"   => [v"1.0", v"1.1",
                    v"2.0", v"2.1", v"2.2",
                    v"3.0", v"3.1", v"3.2",
                    v"4.0", v"4.1", v"4.2",
                    v"5.0", v"5.5",
                    v"6.0", v"6.5",
                    v"7.0", v"7.5",
                    v"8.0",
                    v"9.0", v"9.1", v"9.2",
                    v"10.0", v"10.1"],
    # https://developer.nvidia.com/rdp/cudnn-archive
    "cudnn"     => [v"1.0",
                    v"2.0",
                    v"3.0",
                    v"4.0",
                    v"5.0", v"5.1",
                    v"6.0",
                    v"7.0", v"7.1", v"7.2", v"7.3", v"7.4"]
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
    envdict = Dict(Symbol(var) => ENV[var] for var in envvars if haskey(ENV, var))
    if length(envdict) > 0
        if length(unique(values(envdict))) > 1
            @warn "Multiple CUDA environment variables set to different values" envdict...
        end

        @trace "Looking for CUDA toolkit via environment variables" envdict...
        return collect(values(envdict))
    end

    dirs = String[]

    # look for the compiler binary (in the case PATH points to the installation)
    nvcc_path = find_cuda_binary("nvcc")
    if nvcc_path !== nothing
        nvcc_dir = dirname(nvcc_path)
        if occursin(r"^bin(32|64)?$", basename(nvcc_dir))
            nvcc_dir = dirname(nvcc_dir)
        end

        @trace "Looking for CUDA toolkit via nvcc binary" path=nvcc_path dir=nvcc_dir
        push!(dirs, nvcc_dir)
    end

    # look for the runtime library (in the case LD_LIBRARY_PATH points to the installation)
    libcudart_path = find_cuda_library("cudart")
    if libcudart_path !== nothing
        libcudart_dir = dirname(libcudart_path)
        if occursin(r"^(lib|bin)(32|64)?$", basename(libcudart_dir))
            libcudart_dir = dirname(libcudart_dir)
        end

        @trace "Looking for CUDA toolkit via CUDA runtime library" path=libcudart_path dir=libcudart_dir
        push!(dirs, libcudart_dir)
    end

    # look in default installation directories
    default_dirs = []
    if Sys.iswindows()
        # CUDA versions are installed in separate directories under a single base dir
        program_files = ENV[Sys.WORD_SIZE == 64 ? "ProgramFiles" : "ProgramFiles(x86)" ]
        basedir = joinpath(program_files, "NVIDIA GPU Computing Toolkit", "CUDA")
        if isdir(basedir)
            entries = map(dir -> joinpath(basedir, dir), readdir(basedir))
            reverse!(entries) # we want to search starting from the newest CUDA version
            append!(default_dirs, entries)
        end
    else
        # CUDA versions are installed in unversioned dirs, or suffixed with the version
        basedirs = ["/usr/local/cuda", "/opt/cuda"]
        for dir in basedirs
            append!(default_dirs, "$dir-$(ver.major).$(ver.minor)" for ver in cuda_versions["toolkit"])
        end
        append!(default_dirs, basedirs)
        push!(default_dirs, "/usr/lib/nvidia-cuda-toolkit")
        push!(default_dirs, "/usr/share/cuda")
    end
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

# figure out the CUDA toolkit version (by looking at the `nvcc --version` output)
function find_toolkit_version(toolkit_dirs)
    nvcc_path = find_cuda_binary("nvcc", toolkit_dirs)
    if nvcc_path === nothing
        error("CUDA toolkit at $(join(toolkit_dirs, ", ")) doesn't contain nvcc")
    end

    # parse the nvcc version string
    verstr = withenv("LANG"=>"C") do
        read(`$nvcc_path --version`, String)
    end
    m = match(r"\bV(?<major>\d+).(?<minor>\d+).(?<patch>\d+)\b", verstr)
    m !== nothing || error("could not parse NVCC version info (\"$verstr\")")

    version = VersionNumber(parse(Int, m[:major]),
                            parse(Int, m[:minor]),
                            parse(Int, m[:patch]))
    @debug "CUDA toolkit identified as $version"
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
    @trace "Request to look for libdevice $(join(targets, ", "))" locations=toolkit_dirs

    # figure out locations
    dirs = String[]
    for toolkit_dir in toolkit_dirs
        push!(dirs, toolkit_dir)
        push!(dirs, joinpath(toolkit_dir, "libdevice"))
        push!(dirs, joinpath(toolkit_dir, "nvvm", "libdevice"))
    end

    # filter
    dirs = valid_dirs(dirs)
    @trace "Look for libdevice $(join(targets, ", "))" locations=dirs

    for dir in dirs
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
        if library !== nothing
            @debug "Found unified device library at $library"
            return library
        elseif !isempty(libraries)
            @debug "Found split device libraries at $(join(libraries, ", "))"
            return libraries
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

function find_host_compiler(toolkit_version=nothing)
    if !(Sys.iswindows() || Sys.isapple())
        # Unix-like platforms: find compatible GCC binary
        # enumerate possible names for the gcc binary
        # NOTE: this is coarse, and might list invalid, non-existing versions
        gcc_names = [ "gcc", "cuda-gcc" ]
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
            if gcc_path === nothing
                continue
            end

            # parse the GCC version string
            verstr = withenv("LANG" => "C") do
                readlines(`$gcc_path --version`)[1]
            end
            m = match(Regex("^$(basename(gcc_path)) \\(.*\\) ([0-9.]+)"), verstr)
            if m === nothing
                @warn "Could not parse GCC version info (\"$verstr\"), skipping this compiler."
                continue
            end
            gcc_ver = VersionNumber(m.captures[1])
            @trace "Found GCC $gcc_ver at $gcc_path"

            if toolkit_version === nothing || gcc_supported(gcc_ver, toolkit_version)
                push!(gcc_possibilities, (gcc_path, gcc_ver))
            elseif toolkit_version !== nothing
                @warn "Ignoring $gcc_path v$gcc_ver which isn't supported by CUDA $toolkit_version"
            end
        end

        # select the most recent compiler
        if length(gcc_possibilities) == 0
            error("Could not find a suitable GCC")
        end
        sort!(gcc_possibilities; rev=true, lt=(a, b) -> a[2]<b[2])
        host_compiler, host_version = gcc_possibilities[1]
    elseif Sys.iswindows()
        # Windows: search for MSVC in default locations

        # discover Visual Studio installations
        msvc_paths = String[]
        program_files = ENV[Sys.WORD_SIZE == 64 ? "ProgramFiles(x86)" : "ProgramFiles"]
        ## locate ≥ VS2017
        vswhere_dist = joinpath(program_files, "Microsoft Visual Studio", "Installer", "vswhere.exe")
        vswhere_url = "https://github.com/Microsoft/vswhere/releases/download/2.6.7/vswhere.exe"
        let vswhere = Sys.which("vswhere.exe") !== nothing ? "vswhere.exe" : isfile(vswhere_dist) ? vswhere_dist : download(vswhere_url)
            msvc_cmd_tools_dir = readchomp(`$vswhere -latest -products \* -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`)
            if !isempty(msvc_cmd_tools_dir)
                msvc_build_ver = readchomp(joinpath(msvc_cmd_tools_dir, "VC\\Auxiliary\\Build\\Microsoft.VCToolsVersion.default.txt"))
                arch = Sys.WORD_SIZE == 64 ? "x64" : "x86"
                msvc_path = joinpath(msvc_cmd_tools_dir,"VC\\Tools\\MSVC\\$msvc_build_ver\\bin\\Host$arch\\$arch\\cl.exe")
                @trace "Considering MSVC at $msvc_path located with vswhere"
                push!(msvc_paths, msvc_path)
            end
        end
        ## locate VS2012 to 2014
        let envvars = ["VS140COMNTOOLS", "VS120COMNTOOLS", "VS110COMNTOOLS", "VS100COMNTOOLS"]
            envvars_set = filter(var -> haskey(ENV, var), envvars)
            for var in envvars_set
                val = ENV[var]
                arch = Sys.WORD_SIZE == 64 ? "amd64" : "x86"
                msvc_path = joinpath(dirname(dirname(dirname(val))), "VC", "bin", arch, "cl.exe")
                if isfile(msvc_path)
                    @trace "Considering MSVC at $msvc_path located with environment variable $var"
                    push!(msvc_paths, msvc_path)
                end
            end
        end
        ## look in PATH as well
        let msvc_path = Sys.which("cl.exe")
            if msvc_path !== nothing
                push!(msvc_paths, msvc_path)
            end
        end
        isempty(msvc_paths) && error("No Visual Studio installation found")

        # find MSVC versions
        msvc_list = Dict{VersionNumber,String}()
        for path in msvc_paths
            tmpfile = tempname() # TODO: do this with a pipe
            if !success(pipeline(`$path`, stdout=devnull, stderr=tmpfile))
                @warn "Could not execute $path"
                continue
            end
            verstr = read(tmpfile, String)
            m = match(r"Version\s+(\d+(\.\d+)?(\.\d+)?)"i, verstr)
            if m === nothing
                # depending on the locale, this regex might not match
                m = match(r"\b(\d+(\.\d+)?(\.\d+)?)\b"i, verstr)
            end
            if m === nothing
                @warn "Could not parse Visual Studio version info (\"$verstr\"), skipping this compiler."
                continue
            end
            msvc_ver = VersionNumber(m.captures[1])
            msvc_list[msvc_ver] = path
        end

        # check compiler compatibility
        msvc_path, msvc_ver = nothing, nothing
        if toolkit_version !== nothing
            for ver in sort(collect(keys(msvc_list)), rev=true) # search the highest version first
                if msvc_supported(ver, toolkit_version)
                    msvc_path, msvc_ver = msvc_list[ver], ver
                    break
                else
                    @warn "Ignoring $msvc_path v$msvc_ver which isn't supported by CUDA $toolkit_version"
                end
            end
            if msvc_ver === nothing
                error("None of the available Visual Studio C++ compilers ($(join(keys(msvc_list), ", "))) are compatible with CUDA $toolkit_version")
            end
        else
            # take the first found host, which will be the highest version found
            host_pair = first(sort(collect(msvc_list), by=x->x[1], rev=true))
            msvc_path, msvc_ver = last(host_pair), first(host_pair)
        end

        host_compiler, host_version = msvc_path, msvc_ver
    elseif Sys.isapple()
        # GCC is no longer supported on MacOS so let's just use clang
        # TODO: discovery of all compilers, and version matching against the toolkit
        clang_path = find_binary(["clang"])
        if clang_path === nothing
            error("Could not find clang")
        end
        verstr = read(`$clang_path --version`, String)
        m = match(r"version\s+(\d+(\.\d+)?(\.\d+)?)"i, verstr)
        if m === nothing
            error("Could not parse Clang version info (\"$verstr\")")
        end
        clang_ver = VersionNumber(m[1])

        host_compiler, host_version = clang_path, clang_ver
    end

    @debug "Selected host compiler version $host_version at $host_compiler"
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
    if nvcc_path === nothing
        error("CUDA toolkit at $(join(toolkit_dirs, ", ")) doesn't contain nvcc")
    end
    nvcc_version = toolkit_version

    # find a suitable host compiler
    host_compiler, host_version = find_host_compiler(toolkit_version)

    return Toolchain(nvcc_path, nvcc_version,
                     host_compiler, host_version)
end
