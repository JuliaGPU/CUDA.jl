using NNlib: conv2d, conv2d_grad_x, conv2d_grad_w, maxpool2d, avgpool2d, pool2d, pool2d_grad,
  conv3d, conv3d_grad_x, conv3d_grad_w, maxpool3d, avgpool3d, pool3d, pool3d_grad,
  softmax, ∇softmax, logsoftmax, ∇logsoftmax

@testset "NNlib" begin
  testf(conv2d, rand(100, 100, 3, 1), rand(2, 2, 3, 4))
  testf(conv2d_grad_x, rand(100, 100, 3, 1), rand(2, 2, 3, 4), rand(99, 99, 4, 1))
  testf(conv2d_grad_w, rand(100, 100, 3, 1), rand(2, 2, 3, 4), rand(99, 99, 4, 1))

  testf(conv3d, rand(100, 100, 100, 3, 1), rand(2, 2, 2, 3, 4))
  testf(conv3d_grad_x, rand(100, 100, 100, 3, 1), rand(2, 2, 2, 3, 4), rand(99, 99, 99, 4, 1))
  testf(conv3d_grad_w, rand(100, 100, 100, 3, 1), rand(2, 2, 2, 3, 4), rand(99, 99, 99, 4, 1))

  testf(x -> maxpool2d(x, 2), rand(100, 100, 3, 1))
  testf(x -> avgpool2d(x, 2), rand(100, 100, 3, 1))
  testf((x, dy) -> pool2d_grad(x, pool2d(x), dy), rand(100, 100, 3, 1), rand(50, 50, 3, 1))

  testf(x -> maxpool3d(x, 2), rand(100, 100, 100, 3, 1))
  testf(x -> avgpool3d(x, 2), rand(100, 100, 100, 3, 1))
  testf((x, dy) -> pool3d_grad(x, pool3d(x), dy), rand(100, 100, 100, 3, 1), rand(50, 50, 50, 3, 1))

  for dims in [(5,5), (5,)]
    testf(softmax, rand(dims))
    testf(∇softmax, rand(dims), rand(dims))
    testf(logsoftmax, rand(dims))
    testf(∇logsoftmax, rand(dims), rand(dims))
  end
end
