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

testf(f, xs...) = GPUArrays.TestSuite.compare(f, CuArray, xs...)

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
