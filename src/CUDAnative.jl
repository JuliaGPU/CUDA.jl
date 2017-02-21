__precompile__()

module CUDAnative

using CUDAdrv
using LLVM

# non-exported utility functions
import CUDAdrv: debug, DEBUG, trace, TRACE


ext = joinpath(dirname(@__FILE__), "..", "deps", "ext.jl")
isfile(ext) || error("Unable to load $ext\n\nPlease run Pkg.build(\"CUDAnative\"), and restart Julia.")
include(ext)

include("util.jl")

include("jit.jl")
include("device/array.jl")
include("device/intrinsics.jl") # these files contain generated functions,
include("execution.jl")         # so should get loaded last (JuliaLang/julia#19942)


function __init__()
    if CUDAdrv.version() != cuda_version ||
        LLVM.version() != llvm_version ||
        VersionNumber(Base.libllvm_version) != julia_llvm_version
        error("Your set-up has changed. Please re-run Pkg.build(\"CUDAnative\"), and restart Julia.")
    end

    __init_util__()
end

end
