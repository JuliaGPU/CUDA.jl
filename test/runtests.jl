# Pkg.test runs with --check_bounds=1, forcing all bounds checks.
# This is incompatible with CUDAnative (see JuliaGPU/CUDAnative.jl#98)
if Base.JLOptions().check_bounds == 1
  file = @__FILE__
  run(```
    $(Base.julia_cmd())
    --color=$(Base.have_color ? "yes" : "no")
    --compiled-modules=$(Bool(Base.JLOptions().use_compiled_modules) ? "yes" : "no")
    --startup-file=$(Base.JLOptions().startupfile != 2 ? "yes" : "no")
    --code-coverage=$(["none", "user", "all"][1+Base.JLOptions().code_coverage])
    $(file)
    ```)
  exit()
end

using CuArrays, CUDAnative
using CuArrays: @fix
using Test
using Random
using LinearAlgebra

Random.seed!(1)

import CUDAdrv
## pick the most recent device
global dev = nothing
for newdev in CUDAdrv.devices()
  global dev
    if dev == nothing || CUDAdrv.capability(newdev) > CUDAdrv.capability(dev)
        dev = newdev
    end
end
@info("Testing using device $(CUDAdrv.name(dev))")

CuArrays.allowscalar(false)

function testf(f, xs...; ref=f)
  collect(f(cu.(xs)...)) ≈ collect(ref(xs...))
end


using GPUArrays, GPUArrays.TestSuite

@testset "CuArrays" begin
@testset "GPUArray Testsuite" begin
    TestSuite.run_gpuinterface(CuArray)
    TestSuite.run_base(CuArray)
    TestSuite.run_blas(CuArray)
    TestSuite.run_fft(CuArray)
    TestSuite.run_construction(CuArray)
    TestSuite.run_linalg(CuArray)
    TestSuite.run_mapreduce(CuArray)
    CuArrays.allowscalar(true)
    TestSuite.run_indexing(CuArray)
    CuArrays.allowscalar(false)
end

@testset "Array" begin
  xs = CuArray(2, 3)
  @test xs isa CuArray{Float32, 2}
  @test size(xs) == (2, 3)
  @test collect(CuArray([1 2; 3 4])) == [1 2; 3 4]
  @test collect(cu[1, 2, 3]) == [1, 2, 3]
  @test collect(cu([1, 2, 3])) == [1, 2, 3]
  @test testf(vec, rand(5,3))
end

@testset "Indexing" begin
  @test testf(x -> x[1:2, 2], rand(2,3))
  @test testf(x -> x[[2,1], :], rand(2,3))
end

@testset "PermuteDims" begin
  @test testf(x -> permutedims(x, (2, 1)), rand(2, 3))
  @test testf(x -> permutedims(x, (2, 1, 3)), rand(4, 5, 6))
  @test testf(x -> permutedims(x, (3, 1, 2)), rand(4, 5, 6))
end

@testset "Concat" begin
  @test testf(vcat, ones(5), zeros(5))
  @test testf(hcat, rand(3, 3), rand(3, 3))
  @test testf(vcat, rand(3, 3), rand(3, 3))
  @test testf(hcat, rand(3), rand(3))
  @test testf((a,b) -> cat(a,b; dims=4), rand(3, 4), rand(3, 4))
end

@testset "Broadcast" begin
  @test testf((x)       -> fill!(x, 1),       rand(3,3))
  @test testf((x, y)    -> map(+, x, y),      rand(2, 3), rand(2, 3))
  @test testf((x)      -> CUDAnative.sin.(x), rand(2, 3);
              ref=(x)  -> sin.(x))
  @test testf((x)       -> 2x,                rand(2, 3))
  @test testf((x, y)    -> x .+ y,            rand(2, 3), rand(1, 3))
  @test testf((z, x, y) -> z .= x .+ y,       rand(2, 3), rand(2, 3), rand(2))
end

using ForwardDiff: Dual
using NNlib

@testset "Broadcast Fix" begin
  @test testf(x -> CUDAnative.log.(x), rand(3,3);
              ref=x -> log.(x))
  @test testf((x,xs) -> CUDAnative.log.(x.+xs), 1, rand(3,3);
              ref=(x,xs) -> log.(x.+xs))
  @test testf(x -> @fix(logσ.(x)), rand(5))

  f(x) = @fix logσ.(x)
  ds = Dual.(rand(5),1)
  @test f(ds) ≈ collect(f(CuArray(ds)))
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
  x = cu([1:10;])
  y = x[6:10]
  @test x.buf == y.buf
  @test collect(y) == [6, 7, 8, 9, 10]
  CuArrays._setindex!(y, -1f0, 3)
  @test collect(y) == [6, 7, -1, 9, 10]
  @test collect(x) == [1, 2, 3, 4, 5, 6, 7, -1, 9, 10]
  @test collect(CuMatrix{eltype(y)}(I, 5, 5)*y) == collect(y)
end

@testset "$f! with diagonal $d" for (f, f!) in ((triu, triu!), (tril, tril!)),
                                          d in -2:2
  A = randn(10, 10)
  @test f(A, d) == Array(f!(CuArray(A), d))
end

if CuArrays.cudnn_available()
  include("nnlib.jl")
end
include("blas.jl")
include("solver.jl")
include("fft.jl")
include("rand.jl")

end
