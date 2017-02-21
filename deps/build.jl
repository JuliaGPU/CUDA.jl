# Discover the CUDA and LLVM toolchains, and figure out which devices we can compile for

const ext = joinpath(@__DIR__, "ext.jl")
try
    using CUDAdrv
    using LLVM

    # non-exported utility functions
    import CUDAdrv: debug

    # helper methods for querying version DBs
    # (tool::VersionNumber => devices::Vector{VersionNumber})
    search(db, predicate) =
        Set(Base.Iterators.flatten(valvec for (key,valvec) in db if predicate(key)))

    # figure out which targets are supported by LLVM
    InitializeAllTargets()
    "nvptx" in LLVM.name.(collect(targets())) ||
        error("Your LLVM does not support the NVPTX back-end. Fix this, and rebuild LLVM.jl and CUDAnative.jl.")

    # figure out which devices are supported by LLVM
    llvm_version = LLVM.version()
    info("Using LLVM $llvm_version")
    const llvm_db = [
        v"3.2" => [v"2.0", v"2.1", v"3.0", v"3.5"],
        v"3.5" => [v"5.0"],
        v"3.7" => [v"3.2", v"3.7", v"5.2", v"5.3"],
        v"3.9" => [v"6.0", v"6.1", v"6.2"]
    ]
    llvm_version = LLVM.version()
    llvm_support = search(llvm_db, ver -> ver <= llvm_version)
    isempty(llvm_support) && error("LLVM $llvm_version does not support any compatible device")

    # figure out which devices are supported by CUDA
    cuda_version = CUDAdrv.version()
    info("Using CUDA $cuda_version")
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
    cuda_support = search(cuda_db, ver -> ver <= cuda_version)
    isempty(cuda_support) && error("CUDA $cuda_version does not support any compatible device")

    # check if Julia's LLVM version matches ours
    julia_llvm_version = VersionNumber(Base.libllvm_version)
    info("Using Julia's LLVM $julia_llvm_version")
    if julia_llvm_version != llvm_version
        error("LLVM $llvm_version incompatible with Julia's LLVM $julia_llvm_version")
    end

    # check if we need to rebuild
    if isfile(ext)
        debug("Checking validity of existing ext.jl...")
        @eval module Previous; include($ext); end
        if  Previous.cuda_version       == cuda_version &&
            Previous.llvm_version       == llvm_version &&
            Previous.julia_llvm_version == julia_llvm_version
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
catch ex
    # if anything goes wrong, wipe the existing ext.jl to prevent the package from loading
    rm(ext; force=true)
    rethrow(ex)
end