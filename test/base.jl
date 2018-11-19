using ForwardDiff: Dual
using LinearAlgebra

@testset "GPUArrays test suite" begin
  GPUArrays.test(CuArray)
end

@testset "Memory" begin
  CuArrays.alloc(0)

  @test (CuArrays.@allocated CuArray{Int32}(undef,1)) == 4

  ret, out = @grab_output CuArrays.@time CuArray{Int32}(undef, 1)
  @test isa(ret, CuArray{Int32})
  @test occursin("1 GPU allocation: 4 bytes", out)

  ret, out = @grab_output CuArrays.@time Base.unsafe_wrap(CuArray, Ptr{Int32}(12345678), (2, 3))
  @test isa(ret, CuArray{Int32})
  @test !occursin("GPU allocation", out)
end

@testset "Array" begin
  xs = CuArray{Int}(undef, 2, 3)
  @test collect(CuArray([1 2; 3 4])) == [1 2; 3 4]
  @test collect(cu[1, 2, 3]) == [1, 2, 3]
  @test collect(cu([1, 2, 3])) == [1, 2, 3]
  @test testf(vec, rand(5,3))
  @test cu(1:3) === 1:3

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
  @test (CuArray{Ptr{Cvoid}}(undef, 1) .= C_NULL) == CuArray([C_NULL])
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

@testset "Broadcast Fix" begin
  @test testf(x -> log.(x), rand(3,3))
  @test testf((x,xs) -> log.(x.+xs), Ref(1), rand(3,3))

  if isdefined(CuArrays, :CUDNN)
    using NNlib

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
  x = CuArray{Float64}(undef)
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
  @test testf(x->view(x, :, 1:4, 3), rand(Float32, 5, 4, 3))
  @allowscalar let x = cu(rand(Float32, 5, 4, 3))
    @test_throws BoundsError view(x, :, :, 1:10)

    # Contiguous views should return new CuArray
    @test typeof(view(x, :, 1, 2)) == CuVector{Float32}
    @test typeof(view(x, 1:4, 1, 2)) == CuVector{Float32}
    @test typeof(view(x, :, 1:4, 3)) == CuMatrix{Float32}
    @test typeof(view(x, :, :, 1)) == CuMatrix{Float32}
    @test typeof(view(x, :, :, :)) == CuArray{Float32,3}
    @test typeof(view(x, :)) == CuVector{Float32}
    @test typeof(view(x, 1:3)) == CuVector{Float32}

    # Non-contiguous views should fall back to base's SubArray
    @test typeof(view(x, 1:3, 1:3, 3)) <: SubArray
    @test typeof(view(x, 1, :, 3)) <: SubArray
    @test typeof(view(x, 1, 1:4, 3)) <: SubArray
    @test typeof(view(x, :, 1, 1:3)) <: SubArray
    @test typeof(view(x, :, 1:2:4, 1)) <: SubArray
    @test typeof(view(x, 1:2:5, 1, 1)) <: SubArray
  end
end

@testset "$f! with diagonal $d" for (f, f!) in ((triu, triu!), (tril, tril!)),
                                          d in -2:2
  A = randn(10, 10)
  @test f(A, d) == Array(f!(CuArray(A), d))
end

@testset "Utilities" begin
  t = @elapsed ret = CuArrays.@sync begin
    # TODO: do something that takes a while on the GPU
    #       (need to wrap clock64 in CUDAnative for that)
    42
  end
  @test t >= 0
  @test ret == 42
end
