@testset "ForwardDiff" begin
  using ForwardDiff

  @info "Testing ForwardDiff integration"

  function test_derivative(f, x::T) where T
    buf = CuArray(zeros(T))

    function kernel(a, x)
      a[] = ForwardDiff.derivative(f, x)
      return
    end
    CUDAnative.@cuda kernel(buf, x)
    return buf[]
  end

  testf(cuf, f, x) = test_derivative(cuf, x) ≈ ForwardDiff.derivative(f, x)


  @testset "UNARY" begin
    fs = filter(x->x[1] ==:CUDAnative && x[3] == 1, keys(ForwardDiff.DiffRules.DEFINED_DIFFRULES))


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

    @test testf(x->CUDAnative.pow(x, y), x->x^y, x32)
    @test testf(x->CUDAnative.pow(x, y), x->x^y, x64)
    @test testf(x->CUDAnative.pow(x, y32), x->x^y32, x32)
    @test testf(x->CUDAnative.pow(x, y64), x->x^y64, x64)

    @test testf(y->CUDAnative.pow(x32, y), y->x32^y, y32)
    @test testf(y->CUDAnative.pow(x64, y), y->x64^y, y64)

    @test testf(x->CUDAnative.pow(x, x), x->x^x, x32)
    @test testf(x->CUDAnative.pow(x, x), x->x^x, x64)
  end
end
