using NNlib: conv, ∇conv_data, ∇conv_filter,
  maxpool, meanpool, ∇maxpool, ∇meanpool,
  softmax, ∇softmax, logsoftmax, ∇logsoftmax

info("Testing CuArrays/CUDNN")


@testset "NNlib" begin
  @test testf(conv, rand(10, 10, 3, 1), rand(2, 2, 3, 4))
  @test testf(∇conv_data, rand(9, 9, 4, 1), rand(10, 10, 3, 1), rand(2, 2, 3, 4))
  @test testf(∇conv_filter, rand(9, 9, 4, 1), rand(10, 10, 3, 1), rand(2, 2, 3, 4))

  @test testf(conv, rand(10, 10, 10, 3, 1), rand(2, 2, 2, 3, 4))
  @test testf(∇conv_data, rand(9, 9, 9, 4, 1), rand(10, 10, 10, 3, 1), rand(2, 2, 2, 3, 4))
  @test testf(∇conv_filter, rand(9, 9, 9, 4, 1), rand(10, 10, 10, 3, 1), rand(2, 2, 2, 3, 4))

  @test testf(x -> maxpool(x, (2,2)), rand(10, 10, 3, 1))
  @test testf(x -> meanpool(x, (2,2)), rand(10, 10, 3, 1))
  @test testf((x, dy) -> ∇maxpool(dy, maxpool(x, (2,2)), x, (2,2)), rand(10, 10, 3, 1), rand(5, 5, 3, 1))
  @test testf((x, dy) -> ∇meanpool(dy, meanpool(x, (2,2)), x, (2,2)), rand(10, 10, 3, 1), rand(5, 5, 3, 1))

  @test testf(x -> maxpool(x, (2,2,2)), rand(10, 10, 10, 3, 1))
  @test testf(x -> meanpool(x, (2,2,2)), rand(10, 10, 10, 3, 1))
  @test testf((x, dy) -> ∇maxpool(dy, maxpool(x, (2,2,2)), x, (2,2,2)), rand(10, 10, 10, 3, 1), rand(5, 5, 5, 3, 1))
  @test testf((x, dy) -> ∇meanpool(dy, meanpool(x, (2,2,2)), x, (2,2,2)), rand(10, 10, 10, 3, 1), rand(5, 5, 5, 3, 1))

  for dims in [(5,5), (5,)]
    @test testf(softmax, rand(dims))
    @test testf(∇softmax, rand(dims), rand(dims))
    @test testf(logsoftmax, rand(dims))
    @test testf(∇logsoftmax, rand(dims), rand(dims))
  end
end
