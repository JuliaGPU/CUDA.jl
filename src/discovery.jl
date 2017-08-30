export find_library, find_binary, find_toolkit, find_driver


# library names

const libcuda = Compat.Sys.iswindows() ? "nvcuda" : "cuda"
const libnvml = Compat.Sys.iswindows() ? "nvml" : "nvidia-ml"


# generic stuff

find_library(name, prefix::String="") = find_library(name, [prefix])
function find_library(name, prefixes::Vector{String})
    @debug("Looking for $name library in $prefixes")

    for prefix in prefixes
        # figure out names and locations
        if Compat.Sys.iswindows()
            # location of eg. cudart64_xx.dll or cudart32_xx.dll has to be in PATH env var
            # eg. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\bin
            # (by default, it is set by CUDA toolkit installer)
            tag = Sys.WORD_SIZE == 64 ? "64" : "32"
            names = map(ver->"$name$(tag)_$(ver.major)$(ver.minor)", toolkits)
            if isempty(prefix)
                dirs = []
            else
                dirs = [prefix]
            end
        else
            dir = Sys.WORD_SIZE == 64 ? "lib64" : "lib"
            names = ["lib$name"]
            if isempty(prefix)
                dirs = []
            else
                # we also include "parent/lib" for eg. "/usr/lib"
                dirs = ["$prefix/$dir", "$parent/lib", prefix]
            end
        end

        @trace("Checking for $names in $dirs")
        name = Libdl.find_library(names, dirs)

        if !isempty(name)
            # find the full path of the library
            # NOTE: we could just as well use the result of `find_library,
            # but the user might have run this script with eg. LD_LIBRARY_PATH set
            # so we save the full path in order to always be able to load the correct library
            path = Libdl.dlpath(name)
            @debug("Using $name library at $path")
            return path
        end
    end

    error("Could not find $name library")
end

find_binary(name, prefix::String="") = find_binary(name, [prefix])
function find_binary(name, prefixes::Vector{String})
    @debug("Looking for $name binary in $prefixes")

    for prefix in prefixes
        # figure out names and locations
        if Compat.Sys.iswindows()
            name = "$name.exe"
        end
        if isempty(prefix)
            dirs = split(ENV["PATH"], Compat.Sys.iswindows() ? ';' : ':')
            filter!(path->!isempty(path), dirs)
        else
            dirs = [prefix, joinpath(prefix, "bin")]
        end

        @trace("Checking for $names in $dirs")
        paths = [joinpath(dir, name) for dir in dirs]
        paths = unique(filter(ispath, paths))

        if !isempty(paths)
            path = first(paths)
            @debug("Using $name binary at $path")
            return path
        end
    end

    error("Could not find $name binary")
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
