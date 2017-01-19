__precompile__()

module CUDAnative

using CUDAdrv
using LLVM
:NVPTX in LLVM.API.targets || error("Your LLVM library does not support PTX\nPlease build or install a version of LLVM with the NVPTX back-end enabled, and rebuild LLVM.jl.")

# non-exported utility functions
import CUDAdrv: debug, DEBUG, trace, TRACE


ext = joinpath(dirname(@__FILE__), "..", "deps", "ext.jl")
isfile(ext) || error("Unable to load $ext\n\nPlease re-run Pkg.build(\"CUDAnative\"), and restart Julia.")
include(ext)

include("util.jl")
include("types.jl")

include("jit.jl")
include("execution.jl")
include("device/intrinsics.jl")
include("device/array.jl")


function __init__()
    CUDAdrv.version() != cuda_version && warn("CUDA library has been modified. Please re-run Pkg.build(\"CUDAnative\") and restart Julia.")
    LLVM.version() != llvm_version && warn("CUDA library has been modified. Please re-run Pkg.build(\"CUDAnative\") and restart Julia.")

    __init_util__()
end

end
