using CuArrays, Test

include("util.jl")

using Random
Random.seed!(1)

using CUDAnative
using CUDAdrv

# GPUArrays has a testsuite that isn't part of the main package.
# Include it directly.
import GPUArrays
gpuarrays = pathof(GPUArrays)
gpuarrays_root = dirname(dirname(gpuarrays))
include(joinpath(gpuarrays_root, "test", "testsuite.jl"))

testf(f, xs...; kwargs...) = TestSuite.compare(f, CuArray, xs...; kwargs...)

# pick a suiteable device (by available memory,
# but also by capability if testing needs to be thorough)
candidates = [(device!(dev);
               (dev=dev,
                cap=capability(dev),
                mem=CUDAdrv.available_memory()))
              for dev in devices()]
thorough = parse(Bool, get(ENV, "CI_THOROUGH", "false"))
if thorough
    sort!(candidates, by=x->(x.cap, x.mem))
else
    sort!(candidates, by=x->x.mem)
end
pick = last(candidates)
@info("Testing using device $(name(pick.dev)) (compute capability $(pick.cap), $(Base.format_bytes(pick.mem)) available memory) on CUDA driver $(CUDAdrv.version()) and toolkit $(CUDAnative.version())")
device!(pick.dev)

@testset "CuArrays" begin

# ensure CI is using the requested version
if haskey(ENV, "CI") && haskey(ENV, "JULIA_CUDA_VERSION")
  @test CuArrays.release() == VersionNumber(ENV["JULIA_CUDA_VERSION"])
end

CuArrays.allowscalar(false)

CuArrays.enable_timings()

@testset "GPUArrays test suite" begin
  TestSuite.test(CuArray)
end

include("base.jl")
include("memory.jl")
include("blas.jl")
include("rand.jl")
include("fft.jl")
include("sparse.jl")
include("solver.jl")
include("sparse_solver.jl")
include("dnn.jl")
include("tensor.jl")
include("forwarddiff.jl")
include("nnlib.jl")

if haskey(ENV, "CI")
  CuArrays.memory_status()
  CuArrays.pool_timings()
  CuArrays.alloc_timings()
end

CuArrays.reset_timers!()

end
