using CUDAapi
using CUDAdrv
using LLVM


## auxiliary routines

function llvm_support(version)
    @debug("Using LLVM $version")

    InitializeAllTargets()
    "nvptx" in LLVM.name.(collect(targets())) ||
        error("Your LLVM does not support the NVPTX back-end. Fix this, and rebuild LLVM.jl and CUDAnative.jl")

    target_support = CUDAapi.devices_for_llvm(version)
    @trace("LLVM target support: ", join(target_support, ", "))

    ptx_support = CUDAapi.isas_for_llvm(version)
    @trace("LLVM ISA support: ", join(ptx_support, ", "))
    if VERSION >= v"0.7.0-DEV.1959"
        # JuliaLang/julia#23817 includes a patch with PTX ISA 6.0 support
        push!(ptx_support, v"6.0")
    end
    @trace("LLVM PTX support: ", join(ptx_support, ", "))

    return target_support, ptx_support
end

function cuda_support(driver_version, toolkit_version)
    @debug("Using CUDA driver $driver_version and toolkit $toolkit_version")

    # the toolkit version as reported contains major.minor.patch,
    # but the version number returned by libcuda is only major.minor.
    toolkit_version = VersionNumber(toolkit_version.major, toolkit_version.minor)

    driver_target_support = CUDAapi.devices_for_cuda(driver_version)
    toolkit_target_support = CUDAapi.devices_for_cuda(toolkit_version)
    target_support = driver_target_support ∩ toolkit_target_support
    @trace("CUDA target support: ", join(target_support, ", "))

    driver_ptx_support = CUDAapi.isas_for_cuda(driver_version)
    toolkit_ptx_support = CUDAapi.isas_for_cuda(toolkit_version)
    ptx_support = driver_ptx_support ∩ toolkit_ptx_support
    @trace("CUDA PTX support: ", join(ptx_support, ", "))

    return target_support, ptx_support
end


## discovery routines

# find device library bitcode files
function find_libdevice(targets, parent)
    @debug("Looking for libdevice in $parent")

    dirs = [joinpath(parent, "libdevice"), joinpath(parent, "nvvm", "libdevice")]

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
    if library != nothing
        @debug("Found unified libdevice at $library")
        return library
    elseif !isempty(libraries)
        @debug("Found libdevice for ", join(sort(map(ver->"sm_$(ver.major)$(ver.minor)",
                                            keys(libraries))), ", ", " and "), " at $dir")
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

    config[:julia_version] = VERSION
    config[:julia_llvm_version] = VersionNumber(Base.libllvm_version)

    config[:llvm_version] = LLVM.version()
    llvm_targets, llvm_isas = llvm_support(config[:llvm_version])

    toolkit_path = find_toolkit()
    config[:cuda_toolkit_version] = find_toolkit_version(toolkit_path)

    config[:cuda_driver_version] = CUDAdrv.version()
    cuda_targets, cuda_isas = cuda_support(config[:cuda_driver_version], config[:cuda_toolkit_version])

    config[:target_support] = sort(collect(llvm_targets ∩ cuda_targets))
    @debug("Supported device targets: $(join(sort(config[:target_support]), ", "))")
    isempty(config[:target_support]) && error("Your toolchain does not support any device target")

    config[:ptx_support] = sort(collect(llvm_isas ∩ cuda_isas))
    @debug("Supported PTX ISAs: $(join(sort(config[:ptx_support]), ", "))")
    isempty(config[:target_support]) && error("Your toolchain does not support any PTX ISA")

    # discover other CUDA toolkit artifacts
    config[:libdevice] = find_libdevice(config[:target_support], toolkit_path)
    config[:cuobjdump] = find_binary("cuobjdump", toolkit_path)
    config[:ptxas] = find_binary("ptxas", toolkit_path)


    ## compatibility checks

    LLVM.libllvm_system && error("CUDAnative.jl requires LLVM.jl to be built against Julia's LLVM library, not a system-provided one")

    if config[:julia_version] == v"0.6.1"
        warn("Julia 0.6.1 is not supported, please use 0.6.0 or 0.6.1+ (see #124)")
    end

    if config[:cuda_driver_version] != config[:cuda_toolkit_version]
       warn("CUDA toolkit $(config[:cuda_toolkit_version]) does not match driver $(config[:cuda_driver_version]); this may lead to incompatibilities")
    end

    if config[:cuda_driver_version] >= v"9.0-" && maximum(config[:ptx_support]) < v"6.0"
        warn("CUDA 9.0 requires support for PTX ISA 6.0, which your toolchain does not provide.")
    end

    if config[:julia_llvm_version] != config[:llvm_version]
        error("LLVM $(config[:llvm_version]) incompatible with Julia's LLVM $(config[:julia_llvm_version])")
    end


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
    if VERSION >= v"0.7.0-DEV.1735" ? Base.JLOptions().use_compiled_modules==1 :
                                      Base.JLOptions().use_compilecache==1
        Base.compilecache("CUDAnative")
    end

    return
end

main()
