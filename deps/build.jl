using Compat

using CUDAapi


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


## main

const ext = joinpath(@__DIR__, "ext.jl")
const ext_bak = ext * ".bak"

function main()
    ispath(ext) && mv(ext, ext_bak; remove_destination=true)

    # discover stuff
    driver_path = find_driver()
    libcuda_path = find_library(CUDAapi.libcuda, driver_path)
    libcuda_vendor = "NVIDIA"
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
    if isfile(ext_bak)
        @debug("Checking validity of existing ext.jl")
        @eval module Previous; include($ext_bak); end
        if isdefined(Previous, :libcuda_version) && Previous.libcuda_version == libcuda_version &&
           isdefined(Previous, :libcuda_path)    && Previous.libcuda_path == libcuda_path &&
           isdefined(Previous, :libcuda_vendor)  && Previous.libcuda_vendor == libcuda_vendor
            info("CUDAdrv.jl has already been built for this set-up.")
            mv(ext_bak, ext)
            return
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

    # refresh the compile cache
    # NOTE: we need to do this manually, as the package will load & precompile after
    #       not having loaded a nonexistent ext.jl in the case of a failed build,
    #       causing it not to precompile after a subsequent successful build.
    Base.compilecache("CUDAdrv")

    return
end

main()
