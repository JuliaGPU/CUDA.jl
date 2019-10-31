module CUDAnative

using CUDAapi
using CUDAdrv

using LLVM
using LLVM.Interop

using Adapt
using TimerOutputs
using DataStructures

using Libdl


## global state

# version compatibility
const target_support = Ref{Vector{VersionNumber}}()
const ptx_support = Ref{Vector{VersionNumber}}()

# paths
const libdevice = Ref{Union{String,Dict{VersionNumber,String}}}()
const libcudadevrt = Ref{String}()
const nvdisasm = Ref{String}()
const ptxas = Ref{String}()


## source code includes

include("utils.jl")

# needs to be loaded _before_ the compiler infrastructure, because of generated functions
include(joinpath("device", "tools.jl"))
include(joinpath("device", "pointer.jl"))
include(joinpath("device", "array.jl"))
include(joinpath("device", "cuda.jl"))
include(joinpath("device", "llvm.jl"))
include(joinpath("device", "runtime.jl"))

include("init.jl")

include("compiler.jl")
include("execution.jl")
include("exceptions.jl")
include("reflection.jl")

include("deprecated.jl")


## initialization

verlist(vers) = join(map(ver->"$(ver.major).$(ver.minor)", sort(collect(vers))), ", ", " and ")

function llvm_support(version)
    @debug("Using LLVM v$version")

    # https://github.com/JuliaGPU/CUDAnative.jl/issues/428
    if version >= v"8.0" && VERSION < v"1.3.0-DEV.547"
        error("LLVM 8.0 requires a newer version of Julia")
    end

    InitializeAllTargets()
    haskey(targets(), "nvptx") ||
        error("""
            Your LLVM does not support the NVPTX back-end.

            This is very strange; both the official binaries
            and an unmodified build should contain this back-end.""")

    target_support = sort(collect(CUDAapi.devices_for_llvm(version)))

    ptx_support = CUDAapi.isas_for_llvm(version)
    push!(ptx_support, v"6.0") # JuliaLang/julia#23817
    ptx_support = sort(collect(ptx_support))

    @debug("LLVM supports devices $(verlist(target_support)); PTX $(verlist(ptx_support))")
    return target_support, ptx_support
end

function cuda_support(driver_version, toolkit_version)
    @debug("Using CUDA driver v$driver_version and toolkit v$toolkit_version")

    # the toolkit version as reported contains major.minor.patch,
    # but the version number returned by libcuda is only major.minor.
    toolkit_version = VersionNumber(toolkit_version.major, toolkit_version.minor)
    if toolkit_version > driver_version
        @warn("""CUDA $(toolkit_version.major).$(toolkit_version.minor) is not supported by
                 your driver (which supports up to $(driver_version.major).$(driver_version.minor))""")
    end

    driver_target_support = CUDAapi.devices_for_cuda(driver_version)
    toolkit_target_support = CUDAapi.devices_for_cuda(toolkit_version)
    target_support = sort(collect(driver_target_support ∩ toolkit_target_support))

    driver_ptx_support = CUDAapi.isas_for_cuda(driver_version)
    toolkit_ptx_support = CUDAapi.isas_for_cuda(toolkit_version)
    ptx_support = sort(collect(driver_ptx_support ∩ toolkit_ptx_support))

    @debug("CUDA driver supports devices $(verlist(driver_target_support)); PTX $(verlist(driver_ptx_support))")
    @debug("CUDA toolkit supports devices $(verlist(toolkit_target_support)); PTX $(verlist(toolkit_ptx_support))")

    return target_support, ptx_support
end

function __init__()
    if ccall(:jl_generating_output, Cint, ()) == 1
        # don't initialize when we, or any package that depends on us, is precompiling.
        # this makes it possible to precompile on systems without CUDA,
        # at the expense of using the packages in global scope.
        return
    end

    # compiler barrier to avoid *seeing* `ccall`s to unavailable libraries
    Base.invokelatest(__hidden_init__)
end

function __hidden_init__()
    ## target support

    # LLVM.jl

    llvm_version = LLVM.version()
    llvm_targets, llvm_isas = llvm_support(llvm_version)


    # Julia

    julia_llvm_version = Base.libllvm_version
    if julia_llvm_version != llvm_version
        error("LLVM $llvm_version incompatible with Julia's LLVM $julia_llvm_version")
    end


    # CUDA

    toolkit_dirs = find_toolkit()
    cuda_toolkit_version = find_toolkit_version(toolkit_dirs)
    if cuda_toolkit_version <= v"9"
        @warn "CUDAnative.jl only supports CUDA 9.0 or higher (your toolkit provides CUDA $(version()))"
    end

    cuda_targets, cuda_isas = cuda_support(CUDAdrv.version(), cuda_toolkit_version)

    target_support[] = sort(collect(llvm_targets ∩ cuda_targets))
    isempty(target_support[]) && error("Your toolchain does not support any device target")

    ptx_support[] = sort(collect(llvm_isas ∩ cuda_isas))
    isempty(ptx_support[]) && error("Your toolchain does not support any PTX ISA")

    @debug("CUDAnative supports devices $(verlist(target_support[])); PTX $(verlist(ptx_support[]))")

    let val = find_libdevice(target_support[], toolkit_dirs)
        val === nothing && error("Your CUDA installation does not provide libdevice")
        libdevice[] = val
    end

    let val = find_libcudadevrt(toolkit_dirs)
        val === nothing && error("Your CUDA installation does not provide libcudadevrt")
        libcudadevrt[] = val
    end

    let val = find_cuda_binary("nvdisasm", toolkit_dirs)
        val === nothing && error("Your CUDA installation does not provide the nvdisasm binary")
        nvdisasm[] = val
    end

    let val = find_cuda_binary("ptxas", toolkit_dirs)
        val === nothing && error("Your CUDA installation does not provide the ptxas binary")
        ptxas[] = val
    end


    ## actual initialization

    __init_compiler__()

    CUDAdrv.apicall_hook[] = maybe_initialize
end

end
