using Test

include("util.jl")

using Random
Random.seed!(1)

using CuArrays
using CUDAnative

using GPUArrays
import GPUArrays: allowscalar, @allowscalar

testf(f, xs...; kwargs...) = GPUArrays.TestSuite.compare(f, CuArray, xs...; kwargs...)

allowscalar(false)

@testset "CuArrays" begin

include("base.jl")
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
