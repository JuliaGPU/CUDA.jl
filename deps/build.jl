using CUDAapi
using CUDAdrv
using LLVM


## auxiliary routines

# check support for the LLVM version
function check_llvm(llvm_version)
    @debug("Using LLVM $llvm_version")

    InitializeAllTargets()
    "nvptx" in LLVM.name.(collect(targets())) ||
        error("Your LLVM does not support the NVPTX back-end. Fix this, and rebuild LLVM.jl and CUDAnative.jl")

    llvm_support = CUDAapi.devices_for_llvm(llvm_version)
    isempty(llvm_support) && error("LLVM $llvm_version does not support any compatible device")

    return llvm_support
end

# check support for the CUDA version
function check_cuda(driver_version, toolkit_version)
    @debug("Using CUDA driver $driver_version and toolkit $toolkit_version")

    # the toolkit version as reported contains major.minor.patch,
    # but the version number returned by libcuda is only major.minor.
    toolkit_version = VersionNumber(toolkit_version.major, toolkit_version.minor)

    if driver_version != toolkit_version
       warn("CUDA toolkit version ($toolkit_version) does not match the driver ($driver_version); this may lead to incompatibilities")
    end

    # to be safe, downgrade to the lowest detected version.
    # this assumes libcuda is backwards compatible, which it isn't (yay),
    # but using a too new version leads to more issues (breaks reflection).
    cuda_version = min(driver_version, toolkit_version)
    @debug("Targeting CUDA $cuda_version")

    cuda_support = CUDAapi.devices_for_cuda(cuda_version)
    isempty(cuda_support) && error("CUDA $cuda_version does not support any compatible device")

    return cuda_version, cuda_support
end

# check support for the Julia version
function check_julia(julia_llvm_version, llvm_version, cuda_version)
    @debug("Using Julia's LLVM $julia_llvm_version")

    if cuda_version >= v"9.0-" && VERSION < v"0.7.0-DEV.1959"
        warn("CUDA 9.0 is only supported on Julia 0.7, intra-thread intrinsics (shuffle, vote, ...) might yield wrong results (see #107)")
    end

    if VERSION == v"0.6.1"
        warn("Julia 0.6.1 is not supported, please use 0.6.0 or 0.6.1+ (see #124)")
    end

    if julia_llvm_version != llvm_version
        error("LLVM $llvm_version incompatible with Julia's LLVM $julia_llvm_version")
    end

    return
end


## discovery routines

# find device library bitcode files
function find_libdevice(capabilities, parent)
    @debug("Looking for libdevice in $parent")

    dirs = ["$parent/libdevice", "$parent/nvvm/libdevice"]

    # filter
    @trace("Finding libdevice in $dirs")
    dirs = filter(isdir, unique(dirs))
    if length(dirs) > 1
        warn("Found multiple libdevice locations: ", join(dirs, ", ", " and "))
    elseif isempty(dirs)
        error("Could not find libdevice")
    end

    # select
    dir = first(dirs)
    @trace("Using libdevice at $dir")

    # parse filenames
    libraries = Dict{VersionNumber,String}()
    for cap in capabilities
        path = joinpath(dir, "libdevice.compute_$(cap.major)$(cap.minor).10.bc")
        if isfile(path)
            libraries[cap] = path
        end
    end
    library = nothing
    let path = joinpath(dir, "libdevice.10.bc")
        if isfile(path)
            library = path
        end
    end

    # select
    if library != nothing
        @debug("Found unified libdevice at $library")
        return library
    elseif !isempty(libraries)
        @debug("Found libdevice for $(join(sort(map(ver->"sm_$(ver.major)$(ver.minor)", keys(libraries))), ", ", " and ")) at $dir")
        return libraries
    else
        error("No suitable libdevice found")
    end
end


## main

const config_path = joinpath(@__DIR__, "ext.jl")
const previous_config_path = config_path * ".bak"

function main()
    ispath(config_path) && mv(config_path, previous_config_path; remove_destination=true)
    config = Dict{Symbol,Any}()


    ## gather info

    # check LLVM compatibility
    config[:llvm_version] = LLVM.version()
    llvm_support = check_llvm(config[:llvm_version])

    # discover CUDA toolkit
    toolkit_path = find_toolkit()
    config[:cuda_toolkit_version] = find_toolkit_version(toolkit_path)

    # check CUDA compatibility
    config[:cuda_driver_version] = CUDAdrv.version()
    config[:cuda_version], cuda_support = check_cuda(config[:cuda_driver_version], config[:cuda_toolkit_version])

    # check Julia compatibility
    config[:julia_llvm_version] = VersionNumber(Base.libllvm_version)
    check_julia(config[:julia_llvm_version], config[:llvm_version], config[:cuda_version])

    # check device compatibility
    config[:capabilities] = Vector{VersionNumber}()
    append!(config[:capabilities], llvm_support ∩ cuda_support)
    @debug("Supported capabilities: $(join(sort(config[:capabilities]), ", "))")

    # discover other CUDA toolkit artifacts
    config[:libdevice] = find_libdevice(config[:capabilities], toolkit_path)
    config[:cuobjdump] = find_binary("cuobjdump", toolkit_path)
    config[:ptxas] = find_binary("ptxas", toolkit_path)


    ## (re)generate ext.jl

    function globals(mod)
        all_names = names(mod, true)
        filter(name-> !any(name .== [module_name(mod), Symbol("#eval"), :eval]), all_names)
    end

    if isfile(previous_config_path)
        @debug("Checking validity of existing ext.jl...")
        @eval module Previous; include($previous_config_path); end
        previous_config = Dict{Symbol,Any}(name => getfield(Previous, name)
                                           for name in globals(Previous))

        if config == previous_config
            info("CUDAnative.jl has already been built for this toolchain, no need to rebuild")
            mv(previous_config_path, config_path)
            return
        end
    end

    open(config_path, "w") do fh
        write(fh, "# autogenerated file with properties of the toolchain\n")
        for (key,val) in config
            write(fh, "const $key = $(repr(val))\n")
        end
    end

    # refresh the compile cache
    # NOTE: we need to do this manually, as the package will load & precompile after
    #       not having loaded a nonexistent ext.jl in the case of a failed build,
    #       causing it not to precompile after a subsequent successful build.
    Base.compilecache("CUDAnative")

    return
end

main()
