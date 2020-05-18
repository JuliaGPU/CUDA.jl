@testset "ForwardDiff" begin
  using ForwardDiff
  using ForwardDiff: Dual

  @info "Testing ForwardDiff integration"

  function test_derivative(f, x::T) where T
    buf = CuArray(zeros(T))

    function kernel(a, x)
      a[] = ForwardDiff.derivative(f, x)
      return
    end
    @cuda kernel(buf, x)
    return CUDA.@allowscalar buf[]
  end

  testf(cuf, f, x) = test_derivative(cuf, x) ≈ ForwardDiff.derivative(f, x)


  @testset "UNARY" begin
    fs = filter(x->x[1] ==:CUDA && x[3] == 1, keys(ForwardDiff.DiffRules.DEFINED_DIFFRULES))


    nonneg = [:log, :log1p, :log10, :log2, :sqrt, :acosh]

    for (m, fn, _) ∈ fs
      cuf = @eval $m.$fn
      f = @eval $fn

      x32 = rand(Float32)
      x64 = rand(Float64)
      nx32 = -x32
      nx64 = -x64

      if fn == :acosh
        x32 += 1
        x64 += 1
      end

      @test testf(cuf, f, x32)
      @test testf(cuf, f, x64)

      if fn ∉ nonneg
        @test testf(cuf, f, nx32)
        @test testf(cuf, f, nx64)
      end
    end
  end

  @testset "POW" begin
    x32 = rand(Float32)
    x64 = rand(Float64)
    y32 = rand(Float32)
    y64 = rand(Float64)
    y = Int32(7)

    @test testf(x->CUDA.pow(x, Int32(7)), x->x^y, x32)
    @test testf(x->CUDA.pow(x, y), x->x^y, x64)
    @test testf(x->CUDA.pow(x, y32), x->x^y32, x32)
    @test testf(x->CUDA.pow(x, y64), x->x^y64, x64)

    @test testf(y->CUDA.pow(x32, y), y->x32^y, y32)
    @test testf(y->CUDA.pow(x64, y), y->x64^y, y64)

    @test testf(x->CUDA.pow(x, x), x->x^x, x32)
    @test testf(x->CUDA.pow(x, x), x->x^x, x64)
  end

  @testset "LITERAL_POW" begin
    for x in [rand(Float32, 10), rand(Float64, 10)],
        p in [1, 2, 3, 4, 5]
      @test ForwardDiff.gradient(_x -> sum(_x .^ p), x) ≈ p .* (x .^ (p - 1))
    end
  end

  @testset "Broadcast Fix" begin
    if CUDA.has_cudnn()
      using NNlib

      f(x) = logσ.(x)
      ds = Dual.(rand(5),1)
      @test f(ds) ≈ collect(f(CuArray(ds)))
    end
  end
end
