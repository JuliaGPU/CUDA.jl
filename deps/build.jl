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

        ccall(sym, Cint, $(esc(types)), $(map(esc, args)...))
    end
end

function version(libpath)
    ref = Ref{Cint}()
    status = @apicall(libpath, :cuDriverGetVersion, (Ptr{Cint}, ), ref)
    @assert status == 0
    return VersionNumber(ref[] ÷ 1000, mod(ref[], 100) ÷ 10)
end

function init(libpath, flags=0)
    @apicall(libpath, :cuInit, (Cint, ), flags)
end


## discovery routines

# find CUDA driver library
function find_libcuda()
    libcuda_name = is_windows() ? "nvcuda" : "libcuda"
    libcuda_locations = if haskey(ENV, "CUDA_DRIVER")
            [ENV["CUDA_DRIVER"]]
        else
            # NOTE: no need to look in CUDA toolkit directories here,
            #       as the driver is system-specific and shipped/built independently
            if is_apple()
                ["/usr/local/cuda/lib"]
            else
                String[]
            end
        end
    libcuda = Libdl.find_library(libcuda_name, libcuda_locations)
    isempty(libcuda) && error("Could not find the CUDA driver library (specify the path to $(libcuda_name) using the CUDA_DRIVER environment variable).")

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
    libcuda_path, libcuda_vendor = find_libcuda()
    status = init(libcuda_path)  # see note below
    if status != 0
        # decode some common errors (as we haven't loaded errors.jl yet)
        if status == -1
            error("Building against CUDA driver stubs, which is not supported.")
        elseif status == 100
            error("Initializing CUDA driver failed: no CUDA hardware available (code 100).")
        elseif status == 999
            error("Initializing CUDA driver failed: unknown error (code 999).")
        else
            error("Initializing CUDA driver failed with code $status.")
        end
    end
    libcuda_version = version(libcuda_path)

    # NOTE: initializing the library isn't necessary, but flushes out errors that otherwise
    #       would happen during `version` or, worse, at package load time.

    # check if we need to rebuild
    if isfile(ext)
        debug("Checking validity of existing ext.jl")
        @eval module Previous; include($ext); end
        if isdefined(Previous, :libcuda_version) && Previous.libcuda_version == libcuda_version &&
           isdefined(Previous, :libcuda_path)    && Previous.libcuda_path == libcuda_path &&
           isdefined(Previous, :libcuda_vendor)  && Previous.libcuda_vendor == libcuda_vendor
            info("CUDAdrv.jl has already been built for this set-up.")
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
