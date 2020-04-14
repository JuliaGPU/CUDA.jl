using Test, Base.CoreLogging
import Base.CoreLogging: Info

using CUDAnative, CUDAdrv
using GPUCompiler
import Adapt, LLVM

include("util.jl")

# TODO: also run the tests with CuArrays.jl
const CuArray = CUDAnative.CuHostArray

@testset "CUDAnative" begin

@test CUDAnative.functional(true)

CUDAnative.version()
CUDAnative.release()

# ensure CI is using the requested version
if haskey(ENV, "CI") && haskey(ENV, "JULIA_CUDA_VERSION")
  @test CUDAnative.release() == VersionNumber(ENV["JULIA_CUDA_VERSION"])
end

length(devices()) > 0 || error("The CUDAnative.jl test suite requires a CUDA device")
include("init.jl")
include("pointer.jl")
include("codegen.jl")

capability(device()) >= v"2.0" || error("The CUDAnative.jl test suite requires a CUDA device with compute capability 2.0 or higher")
include("device/codegen.jl")
include("device/execution.jl")
include("device/pointer.jl")
include("device/array.jl")
include("device/cuda.jl")
include("device/wmma.jl")

include("nvtx.jl")

include("examples.jl")

end
