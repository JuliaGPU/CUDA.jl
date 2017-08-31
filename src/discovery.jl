export find_library, find_binary, find_toolkit, find_driver


# library names

const libcuda = Compat.Sys.iswindows() ? "nvcuda" : "cuda"
const libnvml = Compat.Sys.iswindows() ? "nvml" : "nvidia-ml"


# generic stuff

# TODO: make this work like find_library: always search everywhere, but prefix locations priority.
# especially for find_binary.

# wrapper for Libdl.find_library, looking for more names in more locations.
find_library(name, prefix::String) = find_library(name, [prefix])
function find_library(name, prefixes=String[])
    @debug("Looking for $name library in $prefixes")

    # figure out names
    if Compat.Sys.iswindows()
        tag = Sys.WORD_SIZE == 64 ? "64" : "32"
        names = map(ver->"$name$(tag)_$(ver.major)$(ver.minor)", toolkits)
    else
        names = ["lib$name"]
    end

    # figure out locations
    locations = []
    for prefix in prefixes
        push!(locations, prefix)
        push!(locations, joinpath(prefix, "lib"))
        if Sys.WORD_SIZE == 64
            push!(locations, joinpath(prefix, "lib64"))
        end
    end

    @trace("Checking for $names in $locations")
    name = Libdl.find_library(names, locations)
    if isempty(name)
        error("Could not find $name library")
    end

    # find the full path of the library
    # NOTE: we could just as well use the result of `find_library,
    # but the user might have run this script with eg. LD_LIBRARY_PATH set
    # so we save the full path in order to always be able to load the correct library
    path = Libdl.dlpath(name)
    @debug("Using $name library at $path")
    return path
end

# similar to find_library, but for binaries.
# cfr. Libdl.find_library, looks for `name` in `prefix`, then PATH
find_binary(name, prefix::String) = find_binary(name, [prefix])
function find_binary(name, prefixes::Vector{String}=String[])
    @debug("Looking for $name binary in $prefixes")

    # figure out names
    if Compat.Sys.iswindows()
        name = "$name.exe"
    end

    # figure out locations
    locations = []
    for prefix in prefixes
        push!(locations, prefix)
        push!(locations, joinpath(prefix, "bin"))
    end
    let path = ENV["PATH"]
        dirs = split(path, Compat.Sys.iswindows() ? ';' : ':')
        filter!(path->!isempty(path), dirs)
        append!(locations, dirs)
    end

    @trace("Checking for $name in $locations")
    paths = [joinpath(location, name) for location in locations]
    paths = unique(filter(ispath, paths))
    if isempty(paths)
        error("Could not find $name binary")
    end

    path = first(paths)
    @debug("Using $name binary at $path")
    return path
end


# CUDA-specific

function find_toolkit()
    # figure out locations
    dirs = ["/usr/lib/nvidia-cuda-toolkit",
            "/usr/local/cuda",
            "/opt/cuda"]
    ## look for environment variables (taking priority over default values)
    envvars = ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"]
    envvars_set = filter(var -> haskey(ENV, var), envvars)
    if length(envvars_set) > 0
        envvals = unique(map(var->ENV[var], envvars_set))
        if length(envvals) > 1
            warn("Multiple CUDA environment variables set to different values: $(join(envvars_set, ", ", " and "))")
        end
        unshift!(dirs, envvals...)
    end
    ## look for the runtime library (in the case LD_LIBRARY_PATH points to the installation)
    try
        libcudart_path = find_library("cudart")
        dir = dirname(libcudart_path)
        if ismatch(r"^lib(32|64)?$", basename(dir))
            dir = dirname(dir)
        end
        push!(dirs, dir)
    catch ex
        isa(ex, ErrorException) || rethrow(ex)
    end
    ## look for the compiler binary (in the case PATH points to the installation)
    try
        nvcc_path = find_binary("nvcc")
        dir = dirname(nvcc_path)
        if ismatch(r"^bin(32|64)?$", basename(dir))
            dir = dirname(dir)
        end
        push!(dirs, dir)
    catch ex
        isa(ex, ErrorException) || rethrow(ex)
    end

    dirs = filter(isdir, unique(dirs))
    if length(dirs) > 1
        warn("Found multiple CUDA toolkit installations: ", join(dirs, ", ", " and "))
    elseif isempty(dirs)
        error("Could not find CUDA toolkit; specify using CUDA_(dir|HOME|ROOT) environment variable")
    end

    # select
    dir = first(dirs)
    @debug("Using CUDA toolkit at $dir")
    return dir
end

function find_driver()
    # figure out locations
    dirs = String[]
    ## look for the driver library (in the case LD_LIBRARY_PATH points to the installation)
    try
        libcuda_path = find_library(libcuda)
        dir = dirname(libcuda_path)
        if ismatch(r"^lib(32|64)?$", basename(dir))
            dir = dirname(dir)
        end
        push!(dirs, dir)
    catch ex
        isa(ex, ErrorException) || rethrow(ex)
    end
    ## look for the SMI binary (in the case PATH points to the installation)
    try
        nvidiasmi_path = find_binary("nvidia-smi")
        dir = dirname(nvidiasmi_path)
        if ismatch(r"^bin(32|64)?$", basename(dir))
            dir = dirname(dir)
        end
        push!(dirs, dir)
    catch ex
        isa(ex, ErrorException) || rethrow(ex)
    end

    # filter
    dirs = filter(isdir, unique(dirs))
    if length(dirs) > 1
        warn("Found multiple CUDA driver installations: ", join(dirs, ", ", " and "))
    elseif isempty(dirs)
        error("Could not find CUDA driver")
    end

    # select
    dir = first(dirs)
    @debug("Using CUDA driver at $dir")
    return dir
end
