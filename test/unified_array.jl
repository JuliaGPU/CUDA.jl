using LinearAlgebra
import Adapt

@testset "constructors" begin
  xs = CuUnifiedArray{Int}(undef, 2, 3)
  @test collect(CuUnifiedArray([1 2; 3 4])) == [1 2; 3 4]
  @test Base.elsize(xs) == sizeof(Int)
  @test CuUnifiedArray{Int, 2}(xs) === xs
end

@testset "adapt" begin
  A = rand(Float32, 3, 3)
  dA = CuUnifiedArray(A)
  @test Adapt.adapt(Array, dA) == A
  @test Adapt.adapt(CuUnifiedArray, A) isa CuUnifiedArray
  @test Array(Adapt.adapt(CuUnifiedArray, A)) == A
end

@testset "view" begin
  # bug in parentindices conversion
  let x = CuUnifiedArray{Int}(undef, 1, 1)
    x[1,:] .= 42
    @test Array(x)[1,1] == 42
  end

  # performance loss due to Array indices
  let x = CuUnifiedArray{Int}(undef, 1)
    i = [1]
    y = view(x, i)
    @test parent(y) isa CuUnifiedArray
    @test parentindices(y) isa Tuple{CuUnifiedArray}
  end
end

@testset "reshape" begin
  A = [1 2 3 4
       5 6 7 8]
  gA = reshape(CuUnifiedArray(A),1,8)
  _A = reshape(A,1,8)
  _gA = Array(gA)
  @test all(_A .== _gA)
end

@testset "reinterpret" begin
  A = Int32[-1,-2,-3]
  dA = CuUnifiedArray(A)
  dB = reinterpret(UInt32, dA)
  @test reinterpret(UInt32, A) == Array(dB)

  @testset "exception: non-isbits" begin
    local err
    @test try
      reinterpret(Float64, CuUnifiedArray([1,nothing]))
      nothing
    catch err′
      err = err′
    end isa Exception
    @test occursin(
      "does not yet support union bits types",
      sprint(showerror, err))
  end

  @testset "exception: 0-dim" begin
    local err
    @test try
      reinterpret(Int128, CUDA.fill(1f0))
      nothing
    catch err′
      err = err′
    end isa Exception
    @test occursin(
      "cannot reinterpret a zero-dimensional `Float32` array to `Int128` which is of a different size",
      sprint(showerror, err))
  end

  @testset "exception: divisibility" begin
    local err
    @test try
      reinterpret(Int128, CUDA.ones(3))
      nothing
    catch err′
      err = err′
    end isa Exception
    @test occursin(
      "cannot reinterpret an `Float32` array to `Int128` whose first dimension has size `3`.",
      sprint(showerror, err))
  end
end

@testset "accumulate" begin
  for n in (0, 1, 2, 3, 10, 10_000, 16384, 16384+1) # small, large, odd & even, pow2 and not
    @test TestSuite.compare(x->accumulate(+, x), CuUnifiedArray, rand(n))
  end

  # multidimensional
  for (sizes, dims) in ((2,) => 2,
                        (3,4,5) => 2,
                        (1, 70, 50, 20) => 3)
    @test TestSuite.compare(x->accumulate(+, x; dims=dims), CuUnifiedArray, rand(Int, sizes))
  end

  # using initializer
  for (sizes, dims) in ((2,) => 2,
                        (3,4,5) => 2,
                        (1, 70, 50, 20) => 3)
    @test TestSuite.compare(x->accumulate(+, x; dims=dims, init=100.), CuUnifiedArray, rand(Int, sizes))
  end

  # in place
  @test TestSuite.compare(x->(accumulate!(+, x, copy(x)); x), CuUnifiedArray, rand(2))

  # specialized
  @test TestSuite.compare(cumsum,  CuUnifiedArray, rand(2))
  @test TestSuite.compare(cumprod, CuUnifiedArray, rand(2))
end

@testset "broadcasting" begin
  a = CuUnifiedArray(ones(Float32, 10)*2)
  @test sum(a) == 20
  b = CuUnifiedArray(ones(Float32, 10))
  a .+= b
  @test all(Base.Fix1(==,3.0), a)
end

@testset "multi-gpu addition" begin
  N = 25
  D = length(devices())
  a = reshape(CuUnifiedArray(ones(Float32, N*D)*2), (N, D))
  b = reshape(CuUnifiedArray(ones(Float32, N*D)), (N, D))
  c = reshape(CuUnifiedArray(zeros(Float32, N*D)), (N, D))
  @sync begin
      for (gpu, dev) in enumerate(devices())
          @async begin
              device!(dev)
              @views c[:,gpu] .= a[:,gpu] + b[:,gpu]
          end
      end
  end
  device_synchronize()
  @test all(Base.Fix1(==,3.0), c)
end
