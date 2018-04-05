__precompile__()

module CUDAnative

using CUDAapi
using CUDAdrv

using LLVM
using LLVM.Interop

using Libdl

const ext = joinpath(@__DIR__, "..", "deps", "ext.jl")
isfile(ext) || error("CUDAnative.jl has not been built, please run Pkg.build(\"CUDAnative\").")
include(ext)
if !configured
    # default (non-functional) values for critical variables,
    # making it possible to _load_ the package at all times.
    const target_support = [v"2.0"]
    const cuda_driver_version = v"5.5"
end

include("utils.jl")
include("cgutils.jl")
include("pointer.jl")

# needs to be loaded _before_ the compiler infrastructure, because of generated functions
include(joinpath("device", "intrinsics.jl"))
include(joinpath("device", "array.jl"))
include(joinpath("device", "libdevice.jl"))

include("jit.jl")
include("profile.jl")
include("execution.jl")
include("reflection.jl")
include("irvalidation.jl")

include("init.jl")

end
