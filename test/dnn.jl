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

  # Test for agreement between CPU NNlib and CuDNN versions, across a variety of kwargs
  for num_spatial_dims in (2, 3)
    # Initialize data we'll run our tests over
    C_in = 3
    C_out = 4
    batch_size = 1
    x = rand(Float64, repeat([8], num_spatial_dims)..., C_in, batch_size)
    w = rand(Float64, repeat([2], num_spatial_dims)..., C_in, C_out)
    b = rand(Float64, repeat([1], num_spatial_dims)..., C_in, C_out)

    for options in (
      Dict(),
      Dict(:dilation => 2),
      Dict(:flipkernel => true),
      Dict(:stride => 2),
    )
      cdims = DenseConvDims(x, w; options...)
      y = NNlib.conv(x, w, cdims)

      # Test that basic convolution is equivalent across GPU/CPU
      @test testf((x, w) -> NNlib.conv(x, w, cdims), x, w)
      @test testf((y, w) -> ∇conv_data(y, w, cdims), y, w)
      @test testf((x, y) -> ∇conv_filter(x, y, cdims), x, y)

      # Test that we can use an alternative algorithm without dying
      @test_nowarn NNlib.conv!(cu(y), cu(x), cu(w), cdims; algo=1)
      @test_nowarn NNlib.∇conv_data!(cu(x), cu(y), cu(w), cdims; algo=1)
      @test_nowarn NNlib.∇conv_filter!(cu(w), cu(x), cu(y), cdims; algo=1)
    end

    # Test that pooling is equivalent across GPU/CPU
    pdims = PoolDims(x, 2)
    y = maxpool(x, pdims)
    dy = ones(size(y))
    @test testf(x -> maxpool(x, pdims), x)
    @test testf((dy, y, x) -> ∇maxpool(dy, y, x, pdims), dy, y, x)
    @test testf(x -> maxpool(x, pdims), x)
    @test testf((dy, y, x) -> ∇maxpool(dy, y, x, pdims), dy, y, x)
  
    # CPU implementation of ∇conv_bias!
    db = zeros(Float64, 1, 1, 3, 1)
    function CuArrays.CUDNN.∇conv_bias!(db, y)
      db[:] .= sum(y, dims=(1:(ndims(y)-2)))
      return db
    end
    @test testf(CuArrays.CUDNN.∇conv_bias!, db, y)
  end

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
