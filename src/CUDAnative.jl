module CUDAnative

using CUDAapi
using CUDAdrv

using LLVM
using LLVM.Interop

using Adapt
using TimerOutputs
using DataStructures

using Libdl


## discovery

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
        error("""
            CUDA $(toolkit_version.major).$(toolkit_version.minor) is not supported by
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

let
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

    global const cuda_driver_version = CUDAdrv.version()
    cuda_targets, cuda_isas = cuda_support(cuda_driver_version, cuda_toolkit_version)

    global const target_support = sort(collect(llvm_targets ∩ cuda_targets))
    isempty(target_support) && error("Your toolchain does not support any device target")

    global const ptx_support = sort(collect(llvm_isas ∩ cuda_isas))
    isempty(target_support) && error("Your toolchain does not support any PTX ISA")

    @debug("CUDAnative supports devices $(verlist(target_support)); PTX $(verlist(ptx_support))")

    # discover other CUDA toolkit artifacts
    ## required
    global const libdevice = find_libdevice(target_support, toolkit_dirs)
    libdevice === nothing && error("Available CUDA toolchain does not provide libdevice")
    global const libcudadevrt = find_libcudadevrt(toolkit_dirs)
    libcudadevrt === nothing && error("Available CUDA toolchain does not provide libcudadevrt")
    Base.include_dependency(libcudadevrt)
    ## optional
    global const nvdisasm = find_cuda_binary("nvdisasm", toolkit_dirs)
    global const ptxas = find_cuda_binary("ptxas", toolkit_dirs)
end


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

function __init__()
    __init_compiler__()

    # automatic cache file invalidation (when CUDA or LLVM changes)
    # because of the dependency on CUDAdrv.jl and LLVM.jl

    CUDAdrv.apicall_hook[] = maybe_initialize
end

end
