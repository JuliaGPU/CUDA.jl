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
  for package in ("GPUArrays", "CUDAnative")
    match_package(package, branch)
  end
end

using CuArrays

using CUDAnative
import CUDAdrv

using Test
using Random
using LinearAlgebra

macro grab_output(ex)
    quote
        mktemp() do fname, fout
            ret = nothing
            open(fname, "w") do fout
                redirect_stdout(fout) do
                    ret = $(esc(ex))
                end
            end
            ret, read(fname, String)
        end
    end
end

Random.seed!(1)
CuArrays.allowscalar(false)

testf(f, xs...; kwargs...) = GPUArrays.TestSuite.compare(f, CuArray, xs...; kwargs...)

using GPUArrays

@testset "CuArrays" begin

@testset "GPUArrays test suite" begin
  GPUArrays.test(CuArray)
end

@testset "Memory" begin
  CuArrays.alloc(0)

  @test (CuArrays.@allocated CuArray{Int32}()) == 4

  ret, out = @grab_output CuArrays.@time CuArray{Int32}()
  @test isa(ret, CuArray{Int32})
  @test occursin("1 GPU allocation: 4 bytes", out)
end

@testset "Array" begin
  xs = CuArray(2, 3)
  @test collect(CuArray([1 2; 3 4])) == [1 2; 3 4]
  @test collect(cu[1, 2, 3]) == [1, 2, 3]
  @test collect(cu([1, 2, 3])) == [1, 2, 3]
  @test testf(vec, rand(5,3))

  # Check that allowscalar works
  @test_throws ErrorException xs[1]
  @test_throws ErrorException xs[1] = 1
end

@testset "Broadcast" begin
  @test testf((x)       -> fill!(x, 1),  rand(3,3))
  @test testf((x, y)    -> map(+, x, y), rand(2, 3), rand(2, 3))
  @test testf((x)       -> sin.(x),      rand(2, 3))
  @test testf((x)       -> log.(x) .+ 1, rand(2, 3))
  @test testf((x)       -> 2x,           rand(2, 3))
  @test testf((x, y)    -> x .+ y,       rand(2, 3), rand(1, 3))
  @test testf((z, x, y) -> z .= x .+ y,  rand(2, 3), rand(2, 3), rand(2))
  @test (CuArray{Ptr{Cvoid}}(1) .= C_NULL) == CuArray([C_NULL])
  @test CuArray([1,2,3]) .+ CuArray([1.0,2.0,3.0]) == CuArray([2,4,6])

  @eval struct Whatever{T}
      x::Int
  end
  @test Array(Whatever{Int}.(CuArray([1]))) == Whatever{Int}.([1])
end

# https://github.com/JuliaGPU/CUDAnative.jl/issues/223
@testset "Ref Broadcast" begin
  foobar(idx, A) = A[idx]
  @test CuArray([42]) == foobar.(CuArray([1]), Base.RefValue(CuArray([42])))
end

using ForwardDiff: Dual
using NNlib

@testset "Broadcast Fix" begin
  @test testf(x -> log.(x), rand(3,3))
  @test testf((x,xs) -> log.(x.+xs), Ref(1), rand(3,3))

  if isdefined(CuArrays, :CUDNN)
    @test testf(x -> logσ.(x), rand(5))

    f(x) = logσ.(x)
    ds = Dual.(rand(5),1)
    @test f(ds) ≈ collect(f(CuArray(ds)))
  end
end

@testset "Reduce" begin
  @test testf(x -> sum(x, dims=1), rand(2, 3))
  @test testf(x -> sum(x, dims=2), rand(2, 3))
  @test testf(x -> sum(x -> x^2, x, dims=1), rand(2, 3))
  @test testf(x -> prod(x, dims=2), rand(2, 3))

  @test testf(x -> sum(x), rand(2, 3))
  @test testf(x -> prod(x), rand(2, 3))
end

@testset "0D" begin
  x = CuArray{Float64}()
  x .= 1
  @test collect(x)[] == 1
  x /= 2
  @test collect(x)[] == 0.5
end

@testset "Slices" begin
  @test testf(rand(5)) do x
    y = x[2:4]
    y .= 1
    x
  end
  @test testf(rand(5)) do x
    y = view(x, 2:4)
    y .= 1
    x
  end
  let
    x = cu(rand(Float32, 5, 4, 3))
    @test collect(view(x, :, 1:4, 3)) == view(x, :, 1:4, 3)
    @test_throws BoundsError view(x, :, :, 1:10)
    @test typeof(view(x, :, 1:4, 3)) <: CuMatrix # contiguous view
    @test typeof(view(x, 1, 1:4, 3)) <: SubArray # non-contiguous view
  end
end

@testset "$f! with diagonal $d" for (f, f!) in ((triu, triu!), (tril, tril!)),
                                          d in -2:2
  A = randn(10, 10)
  @test f(A, d) == Array(f!(CuArray(A), d))
end

isdefined(CuArrays, :CUDNN)     && include("dnn.jl")
isdefined(CuArrays, :CUBLAS)    && include("blas.jl")
isdefined(CuArrays, :CUSOLVER)  && include("solver.jl")
isdefined(CuArrays, :CUFFT)     && include("fft.jl")
isdefined(CuArrays, :CURAND)    && include("rand.jl")

end
