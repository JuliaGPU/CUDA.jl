using CuArrays
using Base.Test

CuArrays.allowslow(false)

@testset "CuArrays" begin

@testset "Array" begin
  xs = CuArray(2, 3)
  @test xs isa CuArray{Float64, 2}
  @test size(xs) == (2, 3)
  @test collect(CuArray([1 2; 3 4])) == [1 2; 3 4]
  @test collect(cu[1, 2, 3]) == [1, 2, 3]
end

@testset "Indexing" begin
  xs = CuArray(rand(2, 3))
  # For some reason, this fails during Pkg.test() only...
  @test_broken collect(xs[1:2, 2]) == collect(xs)[1:2, 2]
end

@testset "Broadcast" begin
  @test collect(fill!(CuArray(2,2), 1)) == [1 1; 1 1]

  xs = CuArray(rand(2, 3))
  ys = CuArray(rand(2, 3))
  @test collect(map(+, xs, ys)) ≈ map(+, collect(xs), collect(ys))

  @test collect(CuArrays.sin.(xs)) ≈ sin.(collect(xs))
  ys = CuArray(rand(2, 1))
  @test collect(xs .+ ys) ≈ collect(xs) .+ collect(ys)

  zs = CuArray(rand(2, 3))
  zs .= xs .+ ys
  @test collect(zs) ≈ collect(xs .+ ys)
end

@testset "Reduce" begin
  xs = CuArray(rand(2, 3))
  @test collect(sum(xs, 1)) ≈ sum(collect(xs), 1)
  @test collect(sum(xs, 2)) ≈ sum(collect(xs), 2)
  @test collect(sum(x -> x^2, xs, 1)) ≈ sum(x -> x^2, collect(xs), 1)
  @test collect(prod(xs, 1)) ≈ prod(collect(xs), 1)

  @test sum(xs) ≈ sum(collect(xs))
  @test prod(xs) ≈ prod(collect(xs))
end

@testset "BLAS" begin
  xs = rand(5, 5)
  ys = rand(5)
  @test collect(xs * xs) == collect(xs) * collect(xs)
  @test collect(xs * ys) == collect(xs) * collect(ys)
end

end
