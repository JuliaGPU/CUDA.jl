using Compat

# TODO: put in CUDAapi.jl
include(joinpath(dirname(@__DIR__), "src", "util", "logging.jl"))


## API routines

# these routines are the bare minimum we need from the API during build;
# keep in sync with the actual implementations in src/

macro apicall(libpath, fn, types, args...)
    quote
        lib = Libdl.dlopen($(esc(libpath)))
        sym = Libdl.dlsym(lib, $(esc(fn)))

        status = ccall(sym, Cint, $(esc(types)), $(map(esc, args)...))
        status != 0 && error("CUDA error $status calling ", $fn)
    end
end

function version(libpath)
    ref = Ref{Cint}()
    @apicall(libpath, :cuDriverGetVersion, (Ptr{Cint}, ), ref)
    return VersionNumber(ref[] ÷ 1000, mod(ref[], 100) ÷ 10)
end


## discovery routines

# find CUDA toolkit
function find_cuda()
    cuda_envvars = ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"]
    cuda_envvars_set = filter(var -> haskey(ENV, var), cuda_envvars)
    if length(cuda_envvars_set) > 0
        cuda_paths = unique(map(var->ENV[var], cuda_envvars_set))
        if length(unique(cuda_paths)) > 1
            warn("Multiple CUDA path environment variables set: $(join(cuda_envvars_set, ", ", " and ")). ",
                 "Arbitrarily selecting CUDA at $(first(cuda_paths)). ",
                 "To ensure a consistent path, ensure only a single unique CUDA path is set.")
        end
        cuda_path = Nullable(first(cuda_paths))
    else
        cuda_path = Nullable{String}()
    end

    return cuda_path
end

# find CUDA driver library
function find_libcuda()
    libcuda_name = is_windows() ? "nvcuda.dll" : "libcuda"
    libcuda = Libdl.find_library(libcuda_name)
    # NOTE: no need to look in /opt/cuda or /usr/local/cuda here,
    #       as the driver is kernel-specific and should be installed in standard directories
    isempty(libcuda) && error("CUDA driver library cannot be found.")

    # find the full path of the library
    # NOTE: we could just as well use the result of `find_library,
    #       but the user might have run this script with eg. LD_LIBRARY_PATH set
    #       so we save the full path in order to always be able to load the correct library
    libcuda_path = Libdl.dlpath(libcuda)
    debug("Found $libcuda at $libcuda_path")

    # find the library vendor
    libcuda_vendor = "NVIDIA"
    debug("Vendor: $libcuda_vendor")

    return libcuda_path, libcuda_vendor
end


## main

const ext = joinpath(@__DIR__, "ext.jl")

function main()
    # discover stuff
    cuda_path = find_cuda()
    libcuda_path, libcuda_vendor = find_libcuda()
    libcuda_version = version(libcuda_path)

    # check if we need to rebuild
    if isfile(ext)
        debug("Checking validity of existing ext.jl")
        @eval module Previous; include($ext); end
        if isdefined(Previous, :libcuda_version) && Previous.libcuda_version == libcuda_version &&
           isdefined(Previous, :libcuda_path)    && Previous.libcuda_path == libcuda_path &&
           isdefined(Previous, :libcuda_vendor)  && Previous.libcuda_vendor == libcuda_vendor
            info("CUDAdrv.jl has already been built for this CUDA driver, no need to rebuild.")
        end
    end

    # write ext.jl
    open(ext, "w") do fh
        write(fh, """
            const libcuda_path = "$(escape_string(libcuda_path))"
            const libcuda_version = v"$libcuda_version"
            const libcuda_vendor = "$libcuda_vendor"
            """)
    end
    nothing
end

try
    main()
catch ex
    # if anything goes wrong, wipe the existing ext.jl to prevent the package from loading
    rm(ext; force=true)
    rethrow(ex)
end
