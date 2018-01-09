using NNlib: conv2d, softmax, ∇softmax

@testset "NNlib" begin
  for dims in [(5,5), (5,)]
    testf(conv2d, rand(100, 100, 3, 1), rand(2, 2, 3, 4))
    testf(softmax, rand(dims))
    testf(∇softmax, rand(dims), rand(dims))
  end
end
