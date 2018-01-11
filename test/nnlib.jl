using NNlib: conv2d, conv2d_grad_x, conv2d_grad_w, softmax, ∇softmax

@testset "NNlib" begin
  testf(conv2d, rand(100, 100, 3, 1), rand(2, 2, 3, 4))
  testf(conv2d_grad_x, rand(100, 100, 3, 1), rand(2, 2, 3, 4), rand(99, 99, 4, 1))
  testf(conv2d_grad_w, rand(100, 100, 3, 1), rand(2, 2, 3, 4), rand(99, 99, 4, 1))
  for dims in [(5,5), (5,)]
    testf(softmax, rand(dims))
    testf(∇softmax, rand(dims), rand(dims))
  end
end
