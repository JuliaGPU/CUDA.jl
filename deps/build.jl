using CUDAapi
using CUDAdrv
using LLVM


## auxiliary routines

# check support for the LLVM version
function check_llvm()
    llvm_version = LLVM.version()
    @debug("Using LLVM $llvm_version")

    InitializeAllTargets()
    "nvptx" in LLVM.name.(collect(targets())) ||
        error("Your LLVM does not support the NVPTX back-end. Fix this, and rebuild LLVM.jl and CUDAnative.jl")

    llvm_support = CUDAapi.devices_for_llvm(llvm_version)
    isempty(llvm_support) && error("LLVM $llvm_version does not support any compatible device")

    return llvm_version, llvm_support
end

# check support for the CUDA version
function check_cuda()
    cuda_version = CUDAdrv.version()
    @debug("Using CUDA $cuda_version")

    cuda_support = CUDAapi.devices_for_cuda(cuda_version)
    isempty(cuda_support) && error("CUDA $cuda_version does not support any compatible device")

    return cuda_version, cuda_support
end

# check support for the Julia version
function check_julia(llvm_version, cuda_version)
    if cuda_version >= v"9.0-" && VERSION < v"0.7.0-DEV.1959"
        error("CUDA 9.0 is only supported on Julia 0.7")
    end

    if VERSION == v"0.6.1"
        warn("Julia 0.6.1 is buggy and might break CUDAnative.jl (see #124), please use 0.6.0 or 0.6.1+")
    end

    julia_llvm_version = VersionNumber(Base.libllvm_version)
    @debug("Using Julia's LLVM $julia_llvm_version")

    if julia_llvm_version != llvm_version
        error("LLVM $llvm_version incompatible with Julia's LLVM $julia_llvm_version")
    end

    return julia_llvm_version
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

    # check version support
    config[:llvm_version], llvm_support = check_llvm()
    config[:cuda_version], cuda_support = check_cuda()
    config[:julia_llvm_version] = check_julia(config[:llvm_version], config[:cuda_version])

    # figure out supported capabilities
    config[:capabilities] = Vector{VersionNumber}()
    append!(config[:capabilities], llvm_support ∩ cuda_support)
    @debug("Supported capabilities: $(join(sort(config[:capabilities]), ", "))")

    # discover stuff
    toolkit_path = find_toolkit()
    toolkit_version = find_toolkit_version(toolkit_path)
    config[:libdevice] = find_libdevice(config[:capabilities], toolkit_path)
    config[:cuobjdump] = find_binary("cuobjdump", toolkit_path)
    config[:ptxas] = find_binary("ptxas", toolkit_path)

    if toolkit_version.major != config[:cuda_version].major ||
       toolkit_version.minor != config[:cuda_version].minor
       warn("CUDA toolkit version ($toolkit_version) does not match the driver ($(config[:cuda_version])); this may lead to incompatibilities")
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
    Base.compilecache("CUDAnative")

    return
end

main()
