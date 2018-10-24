# CuArrays development often happens in lockstep with other packages, so try to match branches
if haskey(ENV, "GITLAB_CI")
  using Pkg
  function match_package(package, branch)
    try
      Pkg.add(PackageSpec(name=package, rev=String(branch)))
      @info "Installed $package from $branch branch"
    catch ex
      @warn "Could not install $package from $branch branch, trying master" exception=ex
      Pkg.add(PackageSpec(name=package, rev="master"))
      @info "Installed $package from master branch"
    end
  end

  branch = ENV["CI_COMMIT_REF_NAME"]
  for package in ("GPUArrays", "CUDAnative", "NNlib")
    match_package(package, branch)
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

end
