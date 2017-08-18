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
function check_julia(llvm_version)
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

const ext = joinpath(@__DIR__, "ext.jl")
const ext_bak = ext * ".bak"

function main()
    ispath(ext) && mv(ext, ext_bak; remove_destination=true)

    # check version support
    llvm_version, llvm_support = check_llvm()
    cuda_version, cuda_support = check_cuda()
    julia_llvm_version = check_julia(llvm_version)

    # figure out supported capabilities
    capabilities = Vector{VersionNumber}()
    append!(capabilities, llvm_support ∩ cuda_support)
    @debug("Supported capabilities: $(join(sort(capabilities), ", "))")

    # discover stuff
    toolkit_path = find_toolkit()
    libdevice = find_libdevice(capabilities, toolkit_path)
    cuobjdump = find_binary("cuobjdump", toolkit_path)
    ptxas = find_binary("ptxas", toolkit_path)

    # check if we need to rebuild
    if isfile(ext_bak)
        @debug("Checking validity of existing ext.jl...")
        @eval module Previous; include($ext_bak); end
        if  isdefined(Previous, :cuda_version)       && Previous.cuda_version == cuda_version &&
            isdefined(Previous, :llvm_version)       && Previous.llvm_version == llvm_version &&
            isdefined(Previous, :julia_llvm_version) && Previous.julia_llvm_version == julia_llvm_version &&
            isdefined(Previous, :capabilities)       && Previous.capabilities == capabilities &&
            isdefined(Previous, :libdevice)          && Previous.libdevice == libdevice &&
            isdefined(Previous, :cuobjdump)          && Previous.cuobjdump == cuobjdump &&
            isdefined(Previous, :ptxas)              && Previous.ptxas == ptxas
            info("CUDAnative.jl has already been built for this set-up, no need to rebuild")
            mv(ext_bak, ext)
            return
        end
    end

    # write ext.jl
    open(ext, "w") do fh
        write(fh, """
            # Toolchain properties
            const cuda_version = $(repr(cuda_version))
            const llvm_version = $(repr(llvm_version))
            const julia_llvm_version = $(repr(julia_llvm_version))

            const capabilities = $(repr(capabilities))
            const libdevice = $(repr(libdevice))
            const cuobjdump = $(repr(cuobjdump))
            const ptxas = $(repr(ptxas))
            """)
    end

    # refresh the compile cache
    # NOTE: we need to do this manually, as the package will load & precompile after
    #       not having loaded a nonexistent ext.jl in the case of a failed build,
    #       causing it not to precompile after a subsequent successful build.
    Base.compilecache("CUDAnative")

    return
end

main()
