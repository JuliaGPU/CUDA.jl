# development often happens in lockstep with other packages,
# so check-out the master branch of those packages.
using Pkg
if haskey(ENV, "GITLAB_CI")
  for package in ("GPUArrays", "CUDAnative", "NNlib")
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
isdefined(CuArrays, :CUDNN)     && include("dnn.jl")
isdefined(CuArrays, :CUBLAS)    && include("blas.jl")
isdefined(CuArrays, :CUSPARSE)  && include("sparse.jl")
isdefined(CuArrays, :CUSOLVER)  && include("solver.jl")
isdefined(CuArrays, :CUFFT)     && include("fft.jl")
isdefined(CuArrays, :CURAND)    && include("rand.jl")

if isdefined(CuArrays,:CUSPARSE) && isdefined(CuArrays,:CUSOLVER)
  include("sparse_solver.jl")
end

end
