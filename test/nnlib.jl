using NNlib: softmax, ∇softmax

@testset "NNlib" begin
  for dims in [(5,5), (5,)]
    testf(softmax, rand(dims))
    testf(∇softmax, rand(dims), rand(dims))
  end
end
