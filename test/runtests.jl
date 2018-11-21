# development often happens in lockstep with other packages,
# so check-out the master branch of those packages.
using Pkg
if haskey(ENV, "GITLAB_CI")
  for package in ("CUDAapi", "GPUArrays", "CUDAnative", "NNlib")
    Pkg.add(PackageSpec(name=package, rev="master"))
  end
end

using Test

include("util.jl")

using Random
Random.seed!(1)

using CuArrays

using GPUArrays
import GPUArrays: allowscalar, @allowscalar

testf(f, xs...; kwargs...) = GPUArrays.TestSuite.compare(f, CuArray, xs...; kwargs...)

allowscalar(false)

@testset "CuArrays" begin

include("base.jl")
include("dnn.jl")
include("blas.jl")
include("sparse.jl")
include("solver.jl")
include("fft.jl")
include("rand.jl")
include("sparse_solver.jl")

end
