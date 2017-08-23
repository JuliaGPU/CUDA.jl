using CuArrays
using Base.Test

CuArrays.allowslow(false)

function testf(f, xs...)
  @test collect(f(cu.(xs)...)) ≈ collect(f(xs...))
end

@testset "CuArrays" begin

@testset "Array" begin
  xs = CuArray(2, 3)
  @test xs isa CuArray{Float64, 2}
  @test size(xs) == (2, 3)
  @test collect(CuArray([1 2; 3 4])) == [1 2; 3 4]
  @test collect(cu[1, 2, 3]) == [1, 2, 3]
  @test collect(cu([1, 2, 3])) == [1, 2, 3]
end

@testset "Indexing" begin
  testf(x -> x[1:2, 2], rand(2,3))
  testf(x -> x[[2,1], :], rand(2,3))
end

@testset "PermuteDims" begin
  testf(x -> permutedims(x, (2, 1)), rand(2, 3))
  testf(x -> permutedims(x, (2, 1, 3)), rand(4, 5, 6))
end

@testset "Concat" begin
  testf(hcat, rand(3, 3), rand(3, 3))
  testf(vcat, rand(3, 3), rand(3, 3))
end

@testset "Broadcast" begin
  testf((x)       -> fill!(x, 1),  rand(3,3))
  testf((x, y)    -> map(+, x, y), rand(2, 3), rand(2, 3))
  testf((x)       -> sin.(x),      rand(2, 3))
  testf((x, y)    -> x .+ y,       rand(2, 3), rand(1, 3))
  testf((z, x, y) -> z .= x .+ y,  rand(2, 3), rand(2, 3), rand(2))
end

@testset "Reduce" begin
  testf(x -> sum(x, 1), rand(2, 3))
  testf(x -> sum(x, 2), rand(2, 3))
  testf(x -> sum(x -> x^2, x, 1), rand(2, 3))
  testf(x -> prod(x, 2), rand(2, 3))

  testf(x -> sum(x), rand(2, 3))
  testf(x -> prod(x), rand(2, 3))
end

@testset "BLAS" begin
  testf(*, rand(5, 5), rand(5, 5))
  testf(*, rand(5, 5), rand(5))
  testf(A_mul_Bt, rand(5, 5), rand(5, 5))
  testf(At_mul_B, rand(5, 5), rand(5, 5))
end

using NNlib: softmax, ∇softmax

@testset "NNlib" begin
  for dims in [(5,5), (5,)]
    testf(softmax, rand(dims))
    testf(∇softmax, rand(dims), rand(dims))
  end
end

end
