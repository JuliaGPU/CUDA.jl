using CUDAdrv
using LLVM

# TODO: import from CUDAapi.jl
import CUDAdrv: debug


## auxiliary routines

# helper methods for querying version DBs
# (tool::VersionNumber => devices::Vector{VersionNumber})
search(db, predicate) =
    Set(Base.Iterators.flatten(valvec for (key,valvec) in db if predicate(key)))

# database of LLVM versions and the supported devices
const llvm_db = [
    v"3.2" => [v"2.0", v"2.1", v"3.0", v"3.5"],
    v"3.5" => [v"5.0"],
    v"3.7" => [v"3.2", v"3.7", v"5.2", v"5.3"],
    v"3.9" => [v"6.0", v"6.1", v"6.2"]
]

# check support for the LLVM version
function check_llvm()
    llvm_version = LLVM.version()
    info("Using LLVM $llvm_version")

    InitializeAllTargets()
    "nvptx" in LLVM.name.(collect(targets())) ||
        error("Your LLVM does not support the NVPTX back-end. Fix this, and rebuild LLVM.jl and CUDAnative.jl")

    llvm_support = search(llvm_db, ver -> ver <= llvm_version)
    isempty(llvm_support) && error("LLVM $llvm_version does not support any compatible device")

    return llvm_version, llvm_support
end

# database of CUDA versions and the supported devices
const cuda_db = [
    v"4.0" => [v"2.0", v"2.1"],
    v"4.2" => [v"3.0"],
    v"5.0" => [v"3.5"],
    v"6.0" => [v"3.2", v"5.0"],
    v"6.5" => [v"3.7"],
    v"7.0" => [v"5.2"],
    v"7.5" => [v"5.3"],
    v"8.0" => [v"6.0", v"6.1", v"6.2"]
]

# check support for the CUDA version
function check_cuda()
    cuda_version = CUDAdrv.version()
    info("Using CUDA $cuda_version")

    cuda_support = search(cuda_db, ver -> ver <= cuda_version)
    isempty(cuda_support) && error("CUDA $cuda_version does not support any compatible device")

    return cuda_version, cuda_support
end

# check support for the Julia version
function check_julia(llvm_version)
    julia_llvm_version = VersionNumber(Base.libllvm_version)
    info("Using Julia's LLVM $julia_llvm_version")

    if julia_llvm_version != llvm_version
        error("LLVM $llvm_version incompatible with Julia's LLVM $julia_llvm_version")
    end

    return julia_llvm_version
end


## discovery routines

# find CUDA toolkit
function find_cuda()
    # read environment variables
    envvars = ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"]
    envvars_set = filter(var -> haskey(ENV, var), envvars)
    dirs = if length(envvars_set) > 0
        envvals = unique(map(var->ENV[var], envvars_set))
        if length(envvals) > 1
            warn("Multiple CUDA path environment variables set: $(join(envvars_set, ", ", " and "))")
        end
        envvals
    else
        # default values
        ["/usr/lib/nvidia-cuda-toolkit",
         "/usr/local/cuda",
         "/opt/cuda"]
    end

    cuda_paths = unique(filter(isdir, dirs))
    info("Found CUDA at $(join(dirs, ", ", " and "))")
    return cuda_paths
end

# find device library bitcode files
function find_libdevice(cuda_path, capabilities)
    # find the root directory
    dirs = ["$cuda_path/libdevice", "$cuda_path/nvvm/libdevice"]
    dirs = unique(filter(isdir, dirs))
    if isempty(dirs)
        return nothing
    elseif length(dirs) > 1
        warn("Multiple locations found with device code: $(join(dirs, ", ", " and "))")
    end
    dir = first(dirs)

    # discover device library files
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

    if library != nothing
        info("Found unified libdevice at $library")
        return library
    elseif !isempty(libraries)
        info("Found libdevice for $(join(sort(map(ver->"sm_$(ver.major)$(ver.minor)", keys(libraries))), ", ", " and ")) at $dir")
        return libraries
    else
        return nothing
    end
end

function find_binary(cuda_path, name)
    path = joinpath(cuda_path, "bin", name)
    if ispath(path)
        info("Found $name at $path")
        return path
    else
        return nothing
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
    debug("Supported capabilities: $(join(capabilities, ", "))")

    # discover stuff
    cuda_paths = find_cuda()
    if isempty(cuda_paths)
        error("Could not find CUDA toolkit; specify using CUDA_(PATH|HOME|ROOT) environment variable")
    elseif length(cuda_paths) > 1
        warn("Found multiple CUDA installations")
    end
    cuda_path = nothing
    tools = nothing
    for cuda_path in cuda_paths
        tools = [
            find_libdevice(cuda_path, capabilities),
            find_binary(cuda_path, "cuobjdump"),
            find_binary(cuda_path, "ptxas")
        ]
        all(tools .!= nothing) && break
    end
    any(tools .== nothing) && error("Could not find a usable CUDA installation")
    info("Using CUDA at $cuda_path")
    libdevice, cuobjdump, ptxas = tools

    # check if we need to rebuild
    if isfile(ext_bak)
        debug("Checking validity of existing ext.jl...")
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
