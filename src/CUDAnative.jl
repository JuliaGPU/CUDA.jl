__precompile__()

module CUDAnative

using LLVM
using CUDAdrv
import CUDAdrv: debug, DEBUG, trace, TRACE

const ext = joinpath(@__DIR__, "..", "deps", "ext.jl")
if isfile(ext)
    include(ext)
elseif haskey(ENV, "ONLY_LOAD")
    # special mode where the package is loaded without requiring a successful build.
    # this is useful for loading in unsupported environments, eg. Travis + Documenter.jl
    warn("Only loading the package, without activating any functionality.")
else
    error("Unable to load $ext\n\nPlease run Pkg.build(\"CUDAnative\") and restart Julia.")
end

include("jit.jl")
include("profile.jl")
include(joinpath("device", "array.jl"))
include(joinpath("device", "intrinsics.jl")) # these files contain generated functions,
include("execution.jl")                      # so should get loaded late (JuliaLang/julia#19942)
include("reflection.jl")

function __init__()
    haskey(ENV, "ONLY_LOAD") && return

    if CUDAdrv.version() != cuda_version ||
        LLVM.version() != llvm_version ||
        VersionNumber(Base.libllvm_version) != julia_llvm_version
        error("Your set-up has changed. Please re-run Pkg.build(\"CUDAnative\") and restart Julia.")
    end
    init_jit()
end

end
