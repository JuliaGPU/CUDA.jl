@testset "CUDNN" begin

if !isdefined(CuArrays, :CUDNN)
@warn "Not testing CUDNN"
else
using CuArrays.CUDNN
@info "Testing CUDNN $(CUDNN.version())"

@testset "NNlib" begin
  using NNlib
  using NNlib: ∇conv_data, ∇conv_filter,
               maxpool, meanpool, ∇maxpool, ∇meanpool,
               softmax, ∇softmax, logsoftmax, ∇logsoftmax

  @test testf(NNlib.conv, rand(Float64, 10, 10, 3, 1), rand(Float64, 2, 2, 3, 4))
  @test testf(∇conv_data, rand(Float64, 9, 9, 4, 1), rand(Float64, 10, 10, 3, 1), rand(Float64, 2, 2, 3, 4))
  @test testf(∇conv_filter, rand(Float64, 9, 9, 4, 1), rand(Float64, 10, 10, 3, 1), rand(Float64, 2, 2, 3, 4))
  @test testf(CuArrays.CUDNN.∇conv_bias!, cu(rand(Float64, 1, 1, 10, 1)), cu(rand(Float64, 10, 10, 10, 1)))

  @test testf(NNlib.conv, rand(Float64, 10, 10, 3, 1), rand(Float64, 2, 2, 3, 4); dilation=2)
  @test testf(∇conv_data, rand(Float64, 8, 8, 4, 1), rand(Float64, 10, 10, 3, 1), rand(Float64, 2, 2, 3, 4); dilation=2)
  @test testf(∇conv_filter, rand(Float64, 8, 8, 4, 1), rand(Float64, 10, 10, 3, 1), rand(Float64, 2, 2, 3, 4); dilation=2)

  @test testf(NNlib.crosscor, rand(Float64, 10, 10, 3, 1), rand(Float64, 2, 2, 3, 4))
  @test testf(∇conv_data, rand(Float64, 9, 9, 4, 1), rand(Float64, 10, 10, 3, 1), rand(Float64, 2, 2, 3, 4); flipkernel=1)
  @test testf(∇conv_filter, rand(Float64, 9, 9, 4, 1), rand(Float64, 10, 10, 3, 1), rand(Float64, 2, 2, 3, 4); flipkernel=1)

  @test_nowarn NNlib.conv!(cu(zeros(Float64, 9, 9, 3, 1)), cu(rand(Float64, 10, 10, 1, 1)), cu(rand(Float64, 2, 2, 1, 3)), algo=1)
  @test_nowarn NNlib.∇conv_data!(cu(zeros(Float64, 10, 10, 1, 1)), cu(ones(Float64, 9, 9, 3, 1)), cu(rand(Float64, 10, 10, 1, 1)), cu(rand(Float64, 2, 2, 1, 3)), algo=1)
  @test_nowarn NNlib.∇conv_filter!(cu(zeros(Float64, 2, 2, 1, 3)), cu(ones(Float64, 9, 9, 3, 1)), cu(rand(Float64, 10, 10, 1, 1)), cu(rand(Float64, 2, 2, 1, 3)), algo=1)

  @test testf(NNlib.conv, rand(Float64, 10, 10, 10, 3, 1), rand(Float64, 2, 2, 2, 3, 4))
  @test testf(∇conv_data, rand(Float64, 9, 9, 9, 4, 1), rand(Float64, 10, 10, 10, 3, 1), rand(Float64, 2, 2, 2, 3, 4))
  @test testf(∇conv_filter, rand(Float64, 9, 9, 9, 4, 1), rand(Float64, 10, 10, 10, 3, 1), rand(Float64, 2, 2, 2, 3, 4))

  @test testf(NNlib.conv, rand(Float64, 10, 10, 10, 3, 1), rand(Float64, 2, 2, 2, 3, 4); dilation=2)
  @test testf(∇conv_data, rand(Float64, 8, 8, 8, 4, 1), rand(Float64, 10, 10, 10, 3, 1), rand(Float64, 2, 2, 2, 3, 4); dilation=2)
  @test testf(∇conv_filter, rand(Float64, 8, 8, 8, 4, 1), rand(Float64, 10, 10, 10, 3, 1), rand(Float64, 2, 2, 2, 3, 4); dilation=2)

  @test testf(NNlib.crosscor, rand(Float64, 10, 10, 10, 3, 1), rand(Float64, 2, 2, 2, 3, 4))
  @test testf(∇conv_data, rand(Float64, 9, 9, 9, 4, 1), rand(Float64, 10, 10, 10, 3, 1), rand(Float64, 2, 2, 2, 3, 4); flipkernel=1)
  @test testf(∇conv_filter, rand(Float64, 9, 9, 9, 4, 1), rand(Float64, 10, 10, 10, 3, 1), rand(Float64, 2, 2, 2, 3, 4); flipkernel=1)

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

@testset "Activations and Other Ops" begin
  @test testf(CuArrays.CUDNN.cudnnAddTensor, cu(rand(Float64, 10, 10, 3, 1)), cu(rand(Float64, 10, 10, 3, 1)))
  @test testf(CuArrays.CUDNN.cudnnActivationForward, cu(rand(Float64, 10, 10, 3, 1)), cu(rand(Float64, 10, 10, 3, 1)))
  @test testf(CuArrays.CUDNN.cudnnActivationBackward, cu(rand(Float64, 10, 10, 3, 1)), cu(rand(Float64, 10, 10, 3, 1)), cu(rand(Float64, 10, 10, 3, 1)), cu(rand(Float64, 10, 10, 3, 1)))
end

end

end
