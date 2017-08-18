export find_library, find_binary, find_toolkit, find_driver


# library names

const libcuda = is_windows() ? "nvcuda" : "cuda"
const libnvml = is_windows() ? "nvml" : "nvidia-ml"


# generic stuff

function find_library(name, parent=nothing)
    @debug("Looking for $name library in $parent")

    # figure out names and locations
    if is_windows()
        # location of eg. cudart64_xx.dll or cudart32_xx.dll has to be in PATH env var
        # eg. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\bin
        # (by default, it is set by CUDA toolkit installer)
        tag = Sys.WORD_SIZE == 64 ? "64" : "32"
        names = map(ver->"$name$(tag)_$(ver.major)$(ver.minor)", toolkits)
        if parent != nothing
            dirs = [parent]
        else
            dirs = []
        end
    else
        dir = Sys.WORD_SIZE == 64 ? "lib64" : "lib"
        names = ["lib$name"]
        if parent != nothing
            # we also include "parent/lib" for eg. "/usr/lib"
            dirs = ["$parent/$dir", "$parent/lib", parent]
        else
            dirs = []
        end
    end

    @trace("Finding $names in $dirs")
    name = Libdl.find_library(names, dirs)
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

# TODO: look in PATH if parent==nothing
function find_binary(name, parent)
    @debug("Looking for $name binary in $parent")

    # figure out names and locations
    if is_windows()
        name = "$name.exe"
    end
    paths = [joinpath(parent, "bin", name), joinpath(parent, name)]

    @trace("Checking for $names at $paths")
    paths = unique(filter(ispath, paths))
    if length(paths) > 1
        warn("Found multiple $name binaries: ", join(paths, ", ", " and "))
    elseif isempty(paths)
        error("Could not find $name binary")
    end

    # select
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
    end
    ## look for the compiler binary (in the case PATH points to the installation)
    try
        nvcc_path = find_binary("nvcc")
        dir = dirname(nvcc_path)
        if ismatch(r"^bin(32|64)?$", basename(dir))
            dir = dirname(dir)
        end
        push!(dirs, dir)
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
    end
    ## look for the SMI binary (in the case PATH points to the installation)
    try
        nvidiasmi_path = find_binary("nvidia-smi")
        dir = dirname(nvidiasmi_path)
        if ismatch(r"^bin(32|64)?$", basename(dir))
            dir = dirname(dir)
        end
        push!(dirs, dir)
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
