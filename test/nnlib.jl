using NNlib: conv2d, conv2d_grad_x, conv2d_grad_w, maxpool2d, avgpool2d, pool, pool_grad,
  softmax, ∇softmax

@testset "NNlib" begin
  testf(conv2d, rand(100, 100, 3, 1), rand(2, 2, 3, 4))
  testf(conv2d_grad_x, rand(100, 100, 3, 1), rand(2, 2, 3, 4), rand(99, 99, 4, 1))
  testf(conv2d_grad_w, rand(100, 100, 3, 1), rand(2, 2, 3, 4), rand(99, 99, 4, 1))

  testf(x -> maxpool2d(x, 2), rand(100, 100, 3, 1))
  testf(x -> avgpool2d(x, 2), rand(100, 100, 3, 1))
  testf((x, dy) -> pool_grad(x, pool(x), dy), rand(100, 100, 3, 1), rand(50, 50, 3, 1))

  for dims in [(5,5), (5,)]
    testf(softmax, rand(dims))
    testf(∇softmax, rand(dims), rand(dims))
  end
end
