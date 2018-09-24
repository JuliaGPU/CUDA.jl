@testset "cuDNN" begin

@testset "NNlib" begin
  using NNlib: ∇conv_data, ∇conv_filter,
               maxpool, meanpool, ∇maxpool, ∇meanpool,
               softmax, ∇softmax, logsoftmax, ∇logsoftmax

  @test testf(NNlib.conv, rand(Float64, 10, 10, 3, 1), rand(Float64, 2, 2, 3, 4))
  @test testf(∇conv_data, rand(Float64, 9, 9, 4, 1), rand(Float64, 10, 10, 3, 1), rand(Float64, 2, 2, 3, 4))
  @test testf(∇conv_filter, rand(Float64, 9, 9, 4, 1), rand(Float64, 10, 10, 3, 1), rand(Float64, 2, 2, 3, 4))

  @test testf(NNlib.conv, rand(Float64, 10, 10, 10, 3, 1), rand(Float64, 2, 2, 2, 3, 4))
  @test testf(∇conv_data, rand(Float64, 9, 9, 9, 4, 1), rand(Float64, 10, 10, 10, 3, 1), rand(Float64, 2, 2, 2, 3, 4))
  @test testf(∇conv_filter, rand(Float64, 9, 9, 9, 4, 1), rand(Float64, 10, 10, 10, 3, 1), rand(Float64, 2, 2, 2, 3, 4))

  @test testf(x -> maxpool(x, (2,2)), rand(Float64, 10, 10, 3, 1))
  @test testf(x -> meanpool(x, (2,2)), rand(Float64, 10, 10, 3, 1))
  @test testf((x, dy) -> ∇maxpool(dy, maxpool(x, (2,2)), x, (2,2)), rand(Float64, 10, 10, 3, 1), rand(Float64, 5, 5, 3, 1))
  @test testf((x, dy) -> ∇meanpool(dy, meanpool(x, (2,2)), x, (2,2)), rand(Float64, 10, 10, 3, 1), rand(Float64, 5, 5, 3, 1))

  @test testf(x -> maxpool(x, (2,2,2)), rand(Float64, 10, 10, 10, 3, 1))
  @test testf(x -> meanpool(x, (2,2,2)), rand(Float64, 10, 10, 10, 3, 1))
  @test testf((x, dy) -> ∇maxpool(dy, maxpool(x, (2,2,2)), x, (2,2,2)), rand(Float64, 10, 10, 10, 3, 1), rand(Float64, 5, 5, 5, 3, 1))
  @test testf((x, dy) -> ∇meanpool(dy, meanpool(x, (2,2,2)), x, (2,2,2)), rand(Float64, 10, 10, 10, 3, 1), rand(Float64, 5, 5, 5, 3, 1))

  for dims in [(5,5), (5,)]
    @test testf(softmax, rand(Float64, dims))
    @test testf(∇softmax, rand(Float64, dims), rand(Float64, dims))
    @test testf(logsoftmax, rand(Float64, dims))
    @test testf(∇logsoftmax, rand(Float64, dims), rand(Float64, dims))
  end
end

end
