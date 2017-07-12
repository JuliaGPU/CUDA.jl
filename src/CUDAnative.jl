__precompile__()

module CUDAnative

using LLVM
using CUDAdrv
import CUDAdrv: debug, DEBUG, trace, TRACE

const ext = joinpath(@__DIR__, "..", "deps", "ext.jl")
const configured = if isfile(ext)
    include(ext)
    true
else
    # enable CUDAnative.jl to be loaded when the build failed, simplifying downstream use.
    # remove this when we have proper support for conditional modules.
    false
end

include("jit.jl")
include("profile.jl")
include(joinpath("device", "util.jl"))
include(joinpath("device", "array.jl"))
include(joinpath("device", "intrinsics.jl")) # some of these files contain generated functions,
include(joinpath("device", "libdevice.jl"))  # so should get loaded late (JuliaLang/julia#19942)
include("execution.jl")
include("reflection.jl")

const default_device = Ref{CuDevice}()
const default_context = Ref{CuContext}()
function __init__()
    if !configured
        warn("CUDAnative.jl has not been configured, and will not work properly.")
        warn("Please run Pkg.build(\"CUDAnative\") and restart Julia.")
        return
    end

    if CUDAdrv.version() != cuda_version ||
        LLVM.version() != llvm_version ||
        VersionNumber(Base.libllvm_version) != julia_llvm_version
        error("Your set-up has changed. Please run Pkg.build(\"CUDAnative\") and restart Julia.")
    end

    # instantiate a default device and context;
    # this will be implicitly used through `CuCurrentContext`
    # NOTE: although these conceptually match what the primary context is for,
    #       we don't use that because it is refcounted separately
    #       and might confuse / be confused by user operations
    #       (eg. calling `unsafe_reset!` on a primary context)
    default_device[] = CuDevice(0)
    default_context[] = CuContext(default_device[])

    init_jit()
end

end
