using Test, Base.CoreLogging
import Base.CoreLogging: Info

using CUDA, GPUCompiler
import Adapt, LLVM, GPUArrays

# GPUArrays has a testsuite that isn't part of the main package.
# Include it directly.
import GPUArrays
gpuarrays = pathof(GPUArrays)
gpuarrays_root = dirname(dirname(gpuarrays))
include(joinpath(gpuarrays_root, "test", "testsuite.jl"))
testf(f, xs...; kwargs...) = TestSuite.compare(f, CuArray, xs...; kwargs...)

using Random
Random.seed!(1)

@testset "CUDA" begin

include("util.jl")

# ensure CI is using the requested toolkit
if haskey(ENV, "CI") && haskey(ENV, "JULIA_CUDA_VERSION")
  @test CUDA.toolkit_release() == VersionNumber(ENV["JULIA_CUDA_VERSION"])
end

# the order of tests generally follows the order of includes in CUDA.jl

include("initialization.jl")
include("pointer.jl")

# core library
include("apiutils.jl")
include("cuda.jl")

CUDA.allowscalar(false)
CUDA.enable_timings()

# compiler
include("codegen.jl")
include("execution.jl")

# compiler libraries
include("nvtx.jl")

# device functionality
include("device/pointer.jl")
include("device/array.jl")
include("device/cuda.jl")
if VERSION >= v"1.4.1" && capability(device()) >= v"7.0"
include("device/wmma.jl")
end

# array abstraction
include("array.jl")
include("memory.jl")
include("iterator.jl")
@testset "GPUArrays test suite" begin
  TestSuite.test(CuArray)
end

# array libraries
include("cublas.jl")
include("curand.jl")
include("cufft.jl")
include("cusparse.jl")
include("cusolver.jl")
include("cudnn.jl")
include("cutensor.jl")

# integrations
include("forwarddiff.jl")
include("nnlib.jl")
include("statistics.jl")

include("examples.jl")

if haskey(ENV, "CI")
  GC.gc(true)
  CUDA.memory_status()
  CUDA.pool_timings()
  CUDA.alloc_timings()
end

CUDA.reset_timers!()

end
