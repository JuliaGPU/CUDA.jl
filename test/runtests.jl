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

import CuArrays: allowscalar, @allowscalar
allowscalar(false)

CuArrays.enable_timings()

@testset "CuArrays" begin

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

CuArrays.memory_status()
CuArrays.pool_timings()
CuArrays.alloc_timings()

CuArrays.reset_timers!()

end
