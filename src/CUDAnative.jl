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

const toolkit_version = Ref{VersionNumber}()
version() = toolkit_version[]

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
include("device/tools.jl")
include("device/pointer.jl")
include("device/array.jl")
include("device/cuda.jl")
include("device/llvm.jl")
include("device/runtime.jl")

include("init.jl")

include("compiler.jl")
include("execution.jl")
include("exceptions.jl")
include("reflection.jl")

include("deprecated.jl")


## initialization

const __initialized__ = Ref(false)
functional() = __initialized__[]

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0
    silent = parse(Bool, get(ENV, "JULIA_CUDA_SILENT", "false")) || precompiling
    verbose = parse(Bool, get(ENV, "JULIA_CUDA_VERBOSE", "false"))

    # if any dependent GPU package failed, expect it to have logged an error and bail out
    if !CUDAdrv.functional()
        verbose && @warn "CUDAnative.jl did not initialize because CUDAdrv.jl failed to"
        return
    end

    try
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
        toolkit_version[] = find_toolkit_version(toolkit_dirs)
        if toolkit_version[] <= v"9"
            silent || @warn "CUDAnative.jl only supports CUDA 9.0 or higher (your toolkit provides CUDA $(toolkit_version[]))"
        end

        cuda_targets, cuda_isas = cuda_support(CUDAdrv.version(), toolkit_version[])

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

        __initialized__[] = true
    catch ex
        # don't actually fail to keep the package loadable
        if !silent
            if verbose
                @error "CUDAnative.jl failed to initialize" exception=(ex, catch_backtrace())
            else
                @info "CUDAnative.jl failed to initialized, GPU functionality unavailable (set JULIA_CUDA_SILENT or JULIA_CUDA_VERBOSE to silence or expand this message)"
            end
        end
    end
end

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

end
