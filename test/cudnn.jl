using CUDA.CUDNN

@test has_cudnn()
@test CUDNN.version() isa VersionNumber

@testset "NNlib" begin
  using NNlib
  using NNlib: ∇conv_data, ∇conv_filter,
               maxpool, meanpool, ∇maxpool, ∇meanpool,
               softmax, ∇softmax, logsoftmax, ∇logsoftmax
  a, b, c = rand(Float64, 10, 10, 3, 1), rand(Float64, 2, 2, 3, 4), rand(Float64, 9, 9, 4, 1)
  da, db, dc = CuArray(a), CuArray(b), CuArray(c)
  cdims = DenseConvDims(a, b)
  @test NNlib.conv(a, b, cdims) ≈ collect(NNlib.conv(da, db, cdims))
  @test ∇conv_data(c, b, cdims) ≈ collect(∇conv_data(dc, db, cdims))
  @test ∇conv_filter(a, c, cdims) ≈ collect(∇conv_filter(da, dc, cdims))

  # Test for agreement between CPU NNlib and CuDNN versions, across a variety of kwargs
  for num_spatial_dims in (1, 2, 3)
    # Initialize data we'll run our tests over
    C_in = 3
    C_out = 4
    batch_size = 1
    x = rand(Float64, fill(8, num_spatial_dims)..., C_in, batch_size)
    w = rand(Float64, fill(2, num_spatial_dims)..., C_in, C_out)
    b = rand(Float64, fill(1, num_spatial_dims)..., C_in, C_out)
    setups = [
      (kw_conv=(algo=1,)  ,kw_dims=NamedTuple()),
      (kw_conv=(algo=0,)  ,kw_dims=(dilation = 2,)),
      (kw_conv=(algo=1,)  ,kw_dims=(flipkernel = true,)),
      (kw_conv=(algo=1,)  ,kw_dims=(stride = 2,)),

      (kw_conv=(alpha=1,beta=1), kw_dims=NamedTuple()),
      (kw_conv=(alpha=2f0, beta=-5.1f0), kw_dims=(stride=2, dilation=2)),
    ]

    for (kw_conv, kw_dims) in setups
      cdims = DenseConvDims(x, w; kw_dims...)
      y = NNlib.conv(x, w, cdims)

      # Test that basic convolution is equivalent across GPU/CPU
      @test testf((x, w) -> NNlib.conv(x, w, cdims), x, w)
      @test testf((y, w) -> ∇conv_data(y, w, cdims), y, w)
      @test testf((x, y) -> ∇conv_filter(x, y, cdims), x, y)

      # Test that we can use an alternative algorithm without dying
      @test_nowarn NNlib.conv!(CuArray{Float32}(y), CuArray{Float32}(x), CuArray{Float32}(w), cdims; kw_conv...)
      @test_nowarn NNlib.∇conv_data!(CuArray{Float32}(x), CuArray{Float32}(y), CuArray{Float32}(w), cdims; kw_conv...)
      @test_nowarn NNlib.∇conv_filter!(CuArray{Float32}(w), CuArray{Float32}(x), CuArray{Float32}(y), cdims; kw_conv...)
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
    function CUDNN.∇conv_bias!(db, y)
      db .= sum(y, dims=(1:(ndims(y)-2)))
      return db
    end
    #@test testf(CUDNN.∇conv_bias!, db, y)
  end

  for dims in [(5,5), (5,)]
    @test testf(softmax, rand(Float64, dims))
    @test testf(∇softmax, rand(Float64, dims), rand(Float64, dims))
    @test testf(logsoftmax, rand(Float64, dims))
    @test testf(∇logsoftmax, rand(Float64, dims), rand(Float64, dims))
  end
end

@testset "Activations and Other Ops" begin
  @test testf(CUDNN.cudnnAddTensor, CUDA.rand(Float32, 10, 10, 3, 1), CUDA.rand(Float32, 10, 10, 3, 1))
  @test testf(CUDNN.cudnnActivationForward, CUDA.rand(Float32, 10, 10, 3, 1), CUDA.rand(Float32, 10, 10, 3, 1))
  @test testf(CUDNN.cudnnActivationBackward, CUDA.rand(Float32, 10, 10, 3, 1), CUDA.rand(Float32, 10, 10, 3, 1), CUDA.rand(Float32, 10, 10, 3, 1), CUDA.rand(Float32, 10, 10, 3, 1))

  # activations defined in src/nnlib.jl
  ACTIVATION_FUNCTIONS = [σ, logσ, hardσ, hardtanh, relu, leakyrelu, relu6, rrelu,
                          elu, gelu, celu, swish, lisht, selu, trelu, softplus,
                          softsign, logcosh, mish, tanhshrink, softshrink];
  for dims in ((5,5), (5,))
    for f in filter(x -> x != rrelu, ACTIVATION_FUNCTIONS)
      @test testf(x -> f.(x), rand(Float64, dims))
    end
  end

  # softplus does not give `Inf` for large arguments
  x = CuArray([1000.])
  @test all(softplus.(x) .== x)

  # optimized activation overwrote inputs
  let
    x = CUDA.ones(1)
    @test Array(x) == [1f0]
    tanh.(x)
    @test Array(x) == [1f0]
    y = tanh.(x)
    @test Array(x) == [1f0]
    @test Array(y) == [tanh(1f0)]
    x .= tanh.(y)
    @test Array(y) == [tanh(1f0)]
    @test Array(x) == [tanh(tanh(1f0))]
  end
end

@testset "Batchnorm" begin
  v = CUDA.rand(Float32, 2)
  m = CUDA.rand(Float32, 2, 5)
  for training in (false, true)
    CUDNN.batchnorm(v, v, m, v, v, 1.0; training=training)
  end
end
