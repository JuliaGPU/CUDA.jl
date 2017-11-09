__precompile__()

module CUDAnative

using LLVM
using CUDAapi
using CUDAdrv

using Compat

const ext = joinpath(@__DIR__, "..", "deps", "ext.jl")
const configured = if isfile(ext)
    include(ext)
    true
else
    # enable CUDAnative.jl to be loaded when the build failed, simplifying downstream use.
    # remove this when we have proper support for conditional modules.
    const cuda_version = v"5.5"
    false
end

include("cgutils.jl")
include("pointer.jl")

# needs to be loaded _before_ the compiler infrastructure, because of generated functions
include(joinpath("device", "array.jl"))
include(joinpath("device", "intrinsics.jl"))
include(joinpath("device", "libdevice.jl"))

include("jit.jl")
include("profile.jl")
include("execution.jl")
include("reflection.jl")
include("irvalidation.jl")

const initialized = Ref{Bool}(false)
const default_device = Ref{CuDevice}()
const default_context = Ref{CuContext}()
const jlctx = Ref{LLVM.Context}()
function __init__()
    if !configured
        warn("CUDAnative.jl has not been configured, and will not work properly.")
        warn("Please run Pkg.build(\"CUDAnative\") and restart Julia.")
        return
    end

    if CUDAdrv.version() != cuda_driver_version ||
        LLVM.version() != llvm_version ||
        VersionNumber(Base.libllvm_version) != julia_llvm_version
        error("Your set-up has changed. Please run Pkg.build(\"CUDAnative\") and restart Julia.")
    end

    jlctx[] = LLVM.Context(convert(LLVM.API.LLVMContextRef,
                                   cglobal(:jl_LLVMContext, Void)))

    init_jit()

    if haskey(ENV, "_") && basename(ENV["_"]) == "rr"
        warn("Running under rr, which is incompatible with CUDA; disabling initialization.")
    else
        # instantiate a default device and context;
        # this will be implicitly used through `CuCurrentContext`
        # NOTE: although these conceptually match what the primary context is for,
        #       we don't use that because it is refcounted separately
        #       and might confuse / be confused by user operations
        #       (eg. calling `unsafe_reset!` on a primary context)
        initialized[] = true
        default_device[] = CuDevice(0)
        default_context[] = CuContext(default_device[])
    end
end

end
