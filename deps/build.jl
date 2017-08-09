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
        error("Your LLVM does not support the NVPTX back-end. Fix this, and rebuild LLVM.jl and CUDAnative.jl.")

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

# find device library bitcode files
function find_libdevice(cuda_path, supported_capabilities)

    # find the root directory
    if haskey(ENV, "NVVMIR_LIBRARY_DIR")
        dirs = [ENV["NVVMIR_LIBRARY_DIR"]]
    elseif !isnull(cuda_path)
        dirs = ["$(get(cuda_path))/libdevice",
                "$(get(cuda_path))/nvvm/libdevice"]
    else
        dirs = ["/usr/lib/nvidia-cuda-toolkit/libdevice",
                "/usr/local/cuda/nvvm/libdevice",
                "/opt/cuda/nvvm/libdevice"]
    end
    dirs = unique(filter(isdir, dirs))
    if isempty(dirs)
        error("CUDA device library path not found. ",
              "Specify by setting a CUDA path variable, or by setting NVVMIR_LIBRARY_DIR.")
    elseif length(dirs) > 1
        warn("Multiple locations found with device code: $(join(dirs, ", ", " and ")). ",
             "Arbitrarily selecting those at $(first(dirs)).")
    end
    dir = first(dirs)

    # discover device library files
    libraries = Dict{VersionNumber,String}()
    for cap in supported_capabilities
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
        info("Found unified libdevice")
        return library
    elseif !isempty(libraries)
        info("Found libdevice for $(join(sort(map(ver->"sm_$(ver.major)$(ver.minor)", keys(libraries))), ", ", " and "))")
        return libraries
    else
        error("No device libraries found in $dir for your hardware.")
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
    supported_capabilities = Vector{VersionNumber}()
    append!(supported_capabilities, llvm_support ∩ cuda_support)
    debug("Supported capabilities: $(join(supported_capabilities, ", "))")

    # discover stuff
    cuda_path = find_cuda()
    libdevice = find_libdevice(cuda_path, supported_capabilities)

    # check if we need to rebuild
    if isfile(ext_bak)
        debug("Checking validity of existing ext.jl...")
        @eval module Previous; include($ext_bak); end
        if  isdefined(Previous, :cuda_version)           && Previous.cuda_version == cuda_version &&
            isdefined(Previous, :llvm_version)           && Previous.llvm_version == llvm_version &&
            isdefined(Previous, :julia_llvm_version)     && Previous.julia_llvm_version == julia_llvm_version &&
            isdefined(Previous, :supported_capabilities) && Previous.supported_capabilities == supported_capabilities &&
            isdefined(Previous, :libdevice)              && Previous.libdevice == libdevice
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

            const supported_capabilities = $(repr(supported_capabilities))
            const libdevice = $(repr(libdevice))
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
