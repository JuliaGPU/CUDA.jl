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


## main

const ext = joinpath(@__DIR__, "ext.jl")

function main()
    # check version support
    llvm_version, llvm_support = check_llvm()
    cuda_version, cuda_support = check_cuda()
    julia_llvm_version = check_julia(llvm_version)

    # check if we need to rebuild
    if isfile(ext)
        debug("Checking validity of existing ext.jl...")
        @eval module Previous; include($ext); end
        if  isdefined(Previous, :cuda_version)       && Previous.cuda_version == cuda_version &&
            isdefined(Previous, :llvm_version)       && Previous.llvm_version == llvm_version &&
            isdefined(Previous, :julia_llvm_version) && Previous.julia_llvm_version == julia_llvm_version
            info("CUDAnative.jl has already been built for this set-up, no need to rebuild")
            return
        end
    end

    # figure out supported capabilities
    supported_capabilities = Vector{VersionNumber}()
    append!(supported_capabilities, llvm_support ∩ cuda_support)
    debug("Supported capabilities: $(join(supported_capabilities, ", "))")

    # write ext.jl
    open(ext, "w") do fh
        write(fh, """
            # Toolchain properties
            const cuda_version = v"$cuda_version"
            const llvm_version = v"$llvm_version"
            const julia_llvm_version = v"$julia_llvm_version"

            const supported_capabilities = $supported_capabilities
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
