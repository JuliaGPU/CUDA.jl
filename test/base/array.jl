using LinearAlgebra
using ChainRulesCore: add!!, is_inplaceable_destination

@testset "constructors" begin
  let xs = CuArray{Int}(undef, 2, 3)
    # basic properties
    @test device(xs) == device()
    @test context(xs) == context()
    @test collect(CuArray([1 2; 3 4])) == [1 2; 3 4]
    @test collect(cu[1, 2, 3]) == [1, 2, 3]
    @test collect(cu([1, 2, 3])) == [1, 2, 3]
    @test testf(vec, rand(5,3))
    @test cu(1:3) === 1:3
    @test Base.elsize(xs) == sizeof(Int)
    @test pointer(CuArray{Int, 2}(xs)) != pointer(xs)

    # test aggressive conversion to Float32, but only for floats, and only with `cu`
    @test cu([1]) isa CuArray{Int}
    @test cu(Float64[1]) isa CuArray{Float32}
    @test cu(ComplexF64[1+1im]) isa CuArray{ComplexF32}
    @test Adapt.adapt(CuArray, Float64[1]) isa CuArray{Float64}
    @test Adapt.adapt(CuArray, ComplexF64[1]) isa CuArray{ComplexF64}
    @test Adapt.adapt(CuArray{Float16}, Float64[1]) isa CuArray{Float16}
  end

  # test pointer conversions
  let xs = CuVector{Int,CUDA.DeviceMemory}(undef, 1)
    @test_throws ArgumentError Base.unsafe_convert(Ptr{Int}, xs)
    @test_throws ArgumentError Base.unsafe_convert(Ptr{Float32}, xs)
  end

  @test collect(CUDA.zeros(2, 2)) == zeros(Float32, 2, 2)
  @test collect(CUDA.ones(2, 2)) == ones(Float32, 2, 2)

  @test collect(CUDA.fill(0, 2, 2)) == zeros(Float32, 2, 2)
  @test collect(CUDA.fill(1, 2, 2)) == ones(Float32, 2, 2)

  # undef with various forms of dims
  let xs = CuArray{Int, 1, CUDA.DeviceMemory}(undef, (64,))
    @test size(xs) == (64,)
  end
  let xs = CuArray{Int, 1, CUDA.DeviceMemory}()
    @test size(xs) == (0,)
  end
  # cu with too many memory types
  @test_throws ArgumentError("Can only specify one of `device`, `unified`, or `host`") cu([1]; device=true, host=true)
  let
    @gensym typnam
    typ = @eval begin
      struct $typnam
        content::Int
      end
      Base.zero(::Type{$typnam}) = $typnam(1)
      Base.one(::Type{$typnam}) = $typnam(2)
      $typnam
    end
    @test collect(CUDA.zeros(typ, 2, 2)) == zeros(typ, 2, 2)
    @test collect(CUDA.ones(typ, 2, 2)) == ones(typ, 2, 2)
  end
end

mutable struct MyBadType
    a::Any
end
const MyBadType2 = Union{BigFloat, Float32}
struct MyBadType3
    a::MyBadType2
end
@testset "Bad CuArray eltype" begin
    @test_throws ErrorException CuArray{MyBadType, 1}(undef, 64)
    @test_throws ErrorException CuArray{MyBadType2, 1}(undef, 64)
    @test_throws ErrorException CuArray{MyBadType3, 1}(undef, 64)
    @test_throws ErrorException CuArray{BigFloat, 1}(undef, 64)
end

@testset "synchronization" begin
  a = CUDA.zeros(2, 2)
  synchronize(a)
  CUDA.enable_synchronization!(a, false)
  CUDA.enable_synchronization!(a)
end

@testset "unsafe_wrap" begin
    # managed memory -> CuArray
    for a in [cu([1]; device=true), cu([1]; unified=true)]
        p = pointer(a)
        for AT in [CuArray, CuArray{Int}, CuArray{Int,1}, typeof(a)],
            b in [unsafe_wrap(AT, p, 1), unsafe_wrap(AT, p, (1,))]
            @test typeof(b) == typeof(a)
            @test pointer(b) == p
            @test size(b) == (1,)
        end
    end

    # managed memory -> Array
    let a = cu([1]; unified=true)
        p = pointer(a)
        for AT in [Array, Array{Int}, Array{Int,1}],
            b in [unsafe_wrap(AT, p, 1), unsafe_wrap(AT, p, (1,)), unsafe_wrap(AT, a)]
            @test typeof(b) == Array{Int,1}
            @test pointer(b) == reinterpret(Ptr{Int}, p)
            @test size(b) == (1,)
        end
    end
    let a = cu([1]; device=true)
        p = pointer(a)
        @test_throws ArgumentError("Can only create a CPU array object from a unified or host CUDA array") unsafe_wrap(Array, p, 1)
    end

    # unmanaged memory -> CuArray
    # note that the device-side pointer may differ from the host one (i.e., on Tegra)
    let
        # automatic memory selection
        for AT in [CuArray, CuArray{Int}, CuArray{Int,1}],
            f in [a->unsafe_wrap(AT, pointer(a), 1),
                  a->unsafe_wrap(AT, pointer(a), (1,)),
                  a->unsafe_wrap(AT, a)]
            a = [1]
            b = f(a)

            @test typeof(b) <: CuArray{Int,1}
            @test size(b) == (1,)
            @test Array(b) == a
        end

        # host memory
        for AT in [CuArray{Int,1,CUDA.HostMemory}],
            f in [a->unsafe_wrap(AT, pointer(a), 1),
                  a->unsafe_wrap(AT, pointer(a), (1,)),
                  a->unsafe_wrap(AT, a)]
            a = [1]
            b = f(a)

            @test typeof(b) <: CuArray{Int,1,CUDA.HostMemory}
            @test size(b) == (1,)
            @test Array(b) == a
        end

        # unified memory (requires HMM)
        if CUDA.supports_hmm(device())
          for AT in [CuArray{Int,1,CUDA.UnifiedMemory}],
              f in [a->unsafe_wrap(AT, pointer(a), 1),
                    a->unsafe_wrap(AT, pointer(a), (1,)),
                    a->unsafe_wrap(AT, a)]
              a = [1]
              b = f(a)

              @test typeof(b) <: CuArray{Int,1,CUDA.UnifiedMemory}
              @test size(b) == (1,)
              @test Array(b) == a
          end
        end
    end

    # errors
    let a = cu([1]; device=true)
        @test_throws ArgumentError unsafe_wrap(Array, a)
    end
    let a = [1]
        @test_throws ArgumentError unsafe_wrap(CuArray{Int,1,CUDA.DeviceMemory}, a)
    end

    # some actual operations
    let buf = CUDA.alloc(CUDA.HostMemory, sizeof(Int), CUDA.MEMHOSTALLOC_DEVICEMAP)
        gpu_ptr = convert(CuPtr{Int}, buf)
        gpu_arr = unsafe_wrap(CuArray, gpu_ptr, 1)
        gpu_arr .= 42

        synchronize()

        cpu_ptr = convert(Ptr{Int}, buf)
        cpu_arr = unsafe_wrap(Array, cpu_ptr, 1)
        @test cpu_arr == [42]
    end

    # symbols and tuples thereof
    let a = CuArray([:a])
      b = unsafe_wrap(CuArray, pointer(a), 1)
      @test typeof(b) <: CuArray{Symbol,1}
      @test size(b) == (1,)
    end
    let a = CuArray([(:a,:b)])
      b = unsafe_wrap(CuArray, pointer(a), 1)
      @test typeof(b) <: CuArray{Tuple{Symbol,Symbol},1}
      @test size(b) == (1,)
    end
end

@testset "adapt" begin
  A = rand(Float32, 3, 3)
  dA = CuArray(A)
  @test Adapt.adapt(Array, dA) == A
  @test Adapt.adapt(CuArray, A) isa CuArray
  @test Array(Adapt.adapt(CuArray, A)) == A

  @test Adapt.adapt(CuArray{Float64}, A) isa CuArray{Float64}
  @test Adapt.adapt(CuArray{Float64,2}, A) isa CuArray{Float64,2}
  @test Adapt.adapt(CuArray{Float64,2, CUDA.UnifiedMemory}, A) isa CuArray{Float64,2, CUDA.UnifiedMemory}
end

@testset "view" begin
  @test testf(rand(5)) do x
    y = x[2:4]
    y .= 1
    x
  end

  @test testf(rand(5)) do x
    y = view(x, 2:4)
    y .= 1
    x
  end

  @test testf(x->view(x, :, 1:4, 3), rand(Float32, 5, 4, 3))

  let x = CUDA.rand(Float32, 5, 4, 3)
    @test_throws BoundsError view(x, :, :, 1:10)
  end

  # bug in parentindices conversion
  let x = CuArray{Int}(undef, 1, 1)
    x[1,:] .= 42
    @test Array(x)[1,1] == 42
  end

  # bug in conversion of indices (#506)
  show(devnull, cu(view(ones(1), [1])))

  # performance loss due to Array indices
  let x = CuArray{Int}(undef, 1)
    i = [1]
    y = view(x, i)
    @test parent(y) isa CuArray
    @test parentindices(y) isa Tuple{CuArray}
  end

  @testset "GPU array source" begin
      a = rand(3)
      i = rand(1:3, 2)
      @test testf(view, a, i)
      @test testf(view, a, view(i, 2:2))
  end

  @testset "CPU array source" begin
      a = rand(3)
      i = rand(1:3, 2)
      @test testf(view, a, Ref(i))
      @test testf(view, a, Ref(view(i, 2:2)))
  end

  @testset "unmanaged view" begin
    a = CuArray([1,2,3])
    ptr = pointer(a, 2)

    b = unsafe_wrap(CuArray, ptr, 2)
    @test Array(b) == [2,3]

    c = view(b, 2:2)
    @test Array(c) == [3]
  end
end

@testset "reshape" begin
  A = [1 2 3 4
       5 6 7 8]
  gA = reshape(CuArray(A),1,8)
  _A = reshape(A,1,8)
  _gA = Array(gA)
  @test all(_A .== _gA)
  A = [1,2,3,4]
  gA = reshape(CuArray(A),4)

  @testset "unmanaged reshape" begin
    a = CuArray([1,2,3])
    ptr = pointer(a, 2)

    b = unsafe_wrap(CuArray, ptr, 2)
    @test Array(b) == [2,3]

    c = reshape(b, (1,2))
    @test Array(c) == [2 3]
  end
end

@testset "reinterpret" begin
  A = Int32[-1,-2,-3]
  dA = CuArray(A)
  dB = reinterpret(UInt32, dA)
  @test reinterpret(UInt32, A) == Array(dB)

  @test collect(reinterpret(Int32, CUDA.fill(1f0)))[] == reinterpret(Int32, 1f0)

  @testset "unmanaged reinterpret" begin
    a = CuArray(Int32[-1,-2,-3])
    ptr = pointer(a, 2)

    b = unsafe_wrap(CuArray, ptr, 2)
    @test Array(b) == Int32[-2,-3]

    c = reinterpret(UInt32, b)
    @test Array(c) == reinterpret(UInt32, Int32[-2,-3])
  end

  @testset "reinterpret(reshape)" begin
    a = CuArray(ComplexF32[1.0f0+2.0f0*im, 2.0f0im, 3.0f0im])
    b = reinterpret(reshape, Float32, a)
    @test a isa CuArray{ComplexF32, 1}
    @test b isa CuArray{Float32, 2}
    @test Array(b) == [1.0 0.0 0.0; 2.0 2.0 3.0]

    a = CuArray(Float32[1.0 0.0 0.0; 2.0 2.0 3.0])
    b = reinterpret(reshape, ComplexF32, a)
    @test Array(b) == ComplexF32[1.0f0+2.0f0*im, 2.0f0im, 3.0f0im]
  end

  @testset "exception: non-isbits" begin
    local err
    @test try
      reinterpret(Float64, CuArray([1,nothing]))
      nothing
    catch err′
      err = err′
    end isa Exception
    @test occursin(
      "cannot reinterpret an `Union{Nothing, Int64}` array to `Float64`, because not all types are bitstypes",
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


@testset "Dense derivatives" begin
  a = CUDA.rand(Int64, 5, 4, 3)
  @test a isa CuArray

  # Contiguous views should return new CuArray
  @test view(a, :, 1, 2) isa CuVector{Int64}
  @test view(a, 1:4, 1, 2) isa CuVector{Int64}
  @test view(a, :, 1:4, 3) isa CuMatrix{Int64}
  @test view(a, :, :, 1) isa CuMatrix{Int64}
  @test view(a, :, :, :) isa CuArray{Int64,3}
  @test view(a, :) isa CuVector{Int64}
  @test view(a, 1:3) isa CuVector{Int64}
  @test view(a, 1, 1, 1) isa CuArray{Int64}

  # Non-contiguous views should fall back to base's SubArray
  @test view(a, 1:3, 1:3, 3) isa SubArray
  @test view(a, 1, :, 3) isa SubArray
  @test view(a, 1, 1:4, 3) isa SubArray
  @test view(a, :, 1, 1:3) isa SubArray
  @test view(a, :, 1:2:4, 1) isa SubArray
  @test view(a, 1:2:5, 1, 1) isa SubArray

  # CartsianIndices should be treated as scalars
  @test view(a, 1, :, CartesianIndex(3)) isa SubArray
  @test view(a, CartesianIndex(1), :, CartesianIndex(3)) isa SubArray

  b = reshape(a, (6,10))
  @test b isa CuArray
  @test b isa StridedCuArray
  @test view(b, :, :, 1) isa DenseCuArray

  b = reshape(a, :)
  @test b isa CuArray

  b = reinterpret(Float64, a)
  @test b isa CuArray
  @test b isa StridedCuArray
  @test view(b, :, :, 1) isa DenseCuArray
end

@testset "StridedArray" begin
  a = CUDA.rand(Int64, 2,2,2)
  @test a isa StridedCuArray

  @test view(a, :, :, 1) isa StridedCuArray
  @test view(a, :, 1, :) isa StridedCuArray
  @test view(a, 1, :, :) isa StridedCuArray

  b = reshape(a, (2,4))
  @test b isa CuArray
  @test b isa StridedCuArray
  @test view(b, :, 1, :) isa StridedCuArray

  b = reinterpret(Float64, a)
  @test b isa CuArray
  @test b isa StridedCuArray
  @test view(b, :, 1, :) isa StridedCuArray
end

@testset "accumulate" begin
  for n in (0, 1, 2, 3, 10, 10_000, 16384, 16384+1) # small, large, odd & even, pow2 and not
    @test testf(x->accumulate(+, x), rand(n))
    @test testf(x->accumulate(+, x), rand(n,2))
    @test testf((x,y)->accumulate(+, x; init=y), rand(n), rand())
  end

  # multidimensional
  for (sizes, dims) in ((2,) => 2,
                        (3,4,5) => 2,
                        (1, 70, 50, 20) => 3)
    @test testf(x->accumulate(+, x; dims=dims), rand(Int, sizes))
    @test testf(x->accumulate(+, x), rand(Int, sizes))
  end

  # using initializer
  for (sizes, dims) in ((2,) => 2,
                        (3,4,5) => 2,
                        (1, 70, 50, 20) => 3)
    @test testf((x,y)->accumulate(+, x; dims=dims, init=y), rand(Int, sizes), rand(Int))
    @test testf((x,y)->accumulate(+, x; init=y), rand(Int, sizes), rand(Int))
  end

  # in place
  @test testf(x->(accumulate!(+, x, copy(x)); x), rand(2))

  # specialized
  @test testf(cumsum, rand(2))
  @test testf(cumprod, rand(2))

  @test_throws ArgumentError("accumulate does not support the keyword arguments [:bad_kwarg]") accumulate(+, CUDA.rand(1024); bad_kwarg="bad")
end

@testset "logical indexing" begin
  @test CuArray{Int}(undef, 2)[CUDA.ones(Bool, 2)] isa CuArray
  @test CuArray{Int}(undef, 2, 2)[CUDA.ones(Bool, 2, 2)] isa CuArray
  @test CuArray{Int}(undef, 2, 2, 2)[CUDA.ones(Bool, 2, 2, 2)] isa CuArray
  @test CuArray{Int}(undef, 2, 2)[CUDA.ones(Bool, 2), CUDA.ones(Bool, 2)] isa CuArray

  @test CuArray{Int}(undef, 2)[ones(Bool, 2)] isa CuArray
  @test CuArray{Int}(undef, 2, 2)[ones(Bool, 2, 2)] isa CuArray
  @test CuArray{Int}(undef, 2, 2, 2)[ones(Bool, 2, 2, 2)] isa CuArray
  @test CuArray{Int}(undef, 2, 2)[ones(Bool, 2), ones(Bool, 2)] isa CuArray

  @test testf((x,y)->x[y], rand(2), ones(Bool, 2))
  @test testf((x,y)->x[y], rand(2, 2), ones(Bool, 2, 2))
  @test testf((x,y)->x[y], rand(2, 2, 2), ones(Bool, 2, 2, 2))
  @test testf((x,y)->x[y,y], rand(2, 2), ones(Bool, 2))

  @test testf(x -> x[x .> 0.5], rand(2))
  @test testf(x -> x[x .> 0.5], rand(2,2))
  @test testf(x -> x[x .> 0.5], rand(2,2,2))

  @test testf(x -> filter(y->y .> 0.5, x), rand(2))
  @test testf(x -> filter(y->y .> 0.5, x), rand(2,2))
  @test testf(x -> filter(y->y .> 0.5, x), rand(2,2,2))

  A = CuArray([1 2; 3 4; 5 6])
  @test Array(A[CuArray([true, false, true]), :]) == [1 2; 5 6]

  x = CuArray([0.0, 0.25, 0.5, 1.0])
  x[x .> 0] .= 0
  @test Array(x) == zeros(4)
end

@testset "reverse" begin
    # 1-d out-of-place
    @test testf(x->reverse(x), rand(1000))
    @test testf(x->reverse(x, 10), rand(1000))
    @test testf(x->reverse(x, 10, 90), rand(1000))

    # 1-d in-place
    @test testf(x->reverse!(x), rand(1000))
    @test testf(x->reverse!(x, 10), rand(1000))
    @test testf(x->reverse!(x, 10, 90), rand(1000))

    # n-d out-of-place
    for shape in ([1, 2, 4, 3], [4, 2], [5], [2^5, 2^5, 2^5]),
        dim in 1:length(shape)
      @test testf(x->reverse(x; dims=dim), rand(shape...))

      cpu = rand(shape...)
      gpu = CuArray(cpu)
      reverse!(gpu; dims=dim)
      @test Array(gpu) == reverse(cpu; dims=dim)
    end

    # supports multidimensional reverse
    for shape in ([1, 2, 4, 3], [2^5, 2^5, 2^5]),
        dim in ((1,2),(2,3),(1,3),:)
      @test testf(x->reverse(x; dims=dim), rand(shape...))

      cpu = rand(shape...)
      gpu = CuArray(cpu)
      reverse!(gpu; dims=dim)
      @test Array(gpu) == reverse(cpu; dims=dim)
    end

    # wrapped array
    @test testf(x->reverse(x), reshape(rand(2,2), 4))

    # error throwing
    cpu = rand(1,2,3,4)
    gpu = CuArray(cpu)
    @test_throws ArgumentError reverse!(gpu, dims=5)
    @test_throws ArgumentError reverse!(gpu, dims=0)
    @test_throws ArgumentError reverse(gpu, dims=5)
    @test_throws ArgumentError reverse(gpu, dims=0)
end

@testset "findall" begin
    # 1D
    @test testf(x->findall(x), rand(Bool, 0))
    @test testf(x->findall(x), rand(Bool, 100))
    @test testf(x->findall(y->y>0.5, x), rand(100))

    # ND
    let x = rand(Bool, 0, 0)
      @test findall(x) == Array(findall(CuArray(x)))
    end
    let x = rand(Bool, 10, 10)
      @test findall(x) == Array(findall(CuArray(x)))
    end
    let x = rand(10, 10)
      @test findall(y->y>0.5, x) == Array(findall(y->y>0.5, CuArray(x)))
    end
end

@testset "issue #543" begin
  x = CUDA.rand(ComplexF32, 1)
  @test x isa CuArray{Complex{Float32}}

  y = exp.(x)
  @test y isa CuArray{Complex{Float32}}
end

@testset "resizing" begin
  # 1) small arrays (<=10 MiB): should still use doubling policy
  a = CuArray([1, 2, 3])

  # reallocation (add less than half)
  CUDA.resize!(a, 4)
  @test length(a) == 4
  @test Array(a)[1:3] == [1, 2, 3]
  @test a.maxsize == max(4, 2*3) * sizeof(eltype(a))

  # no reallocation 
  CUDA.resize!(a, 5)
  @test length(a) == 5
  @test Array(a)[1:3] == [1, 2, 3]
  @test a.maxsize == 6 * sizeof(eltype(a))

  # reallocation (add more than half)
  CUDA.resize!(a, 12)
  @test length(a) == 12
  @test Array(a)[1:3] == [1, 2, 3]
  @test a.maxsize == max(12, 2*5) * sizeof(eltype(a))

  # 2) large arrays (>10 MiB): should use 1 MiB increments
  b = CUDA.fill(1, 2*1024^2)
  maxsize = b.maxsize

  # should bump by exactly 1 MiB
  CUDA.resize!(b, 2*1024^2 + 1)
  @test length(b) == 2*1024^2 + 1
  @test b.maxsize == maxsize + CUDA.RESIZE_INCREMENT
  @test all(Array(b)[1:2*1024^2] .== 1)

  b = CUDA.fill(1, 2*1024^2)
  maxsize = b.maxsize

  # should bump greater than 1 MiB
  new = CUDA.RESIZE_INCREMENT ÷ sizeof(eltype(b))  
  CUDA.resize!(b, 2*1024^2 + new + 1)
  @test length(b) == 2*1024^2 + new + 1
  @test b.maxsize > maxsize + CUDA.RESIZE_INCREMENT
  @test all(Array(b)[1:2*1024^2] .== 1)

  b = CUDA.fill(1, 2*1024^2)
  maxsize = b.maxsize

  # no reallocation
  CUDA.resize!(b, 2*1024^2 - 1)
  @test length(b) == 2*1024^2 - 1
  @test b.maxsize == maxsize
  @test all(Array(b)[1:2*1024^2 - 1] .== 1)

  # 3) corner cases
  c = CuArray{Int}(undef, 0)
  @test length(c) == 0
  CUDA.resize!(c, 1)
  @test length(c) == 1
  @test c.maxsize == 1 * sizeof(eltype(c))

  c = CuArray{Int}(undef, 1)
  @test length(c) == 1
  CUDA.resize!(c, 0)
  @test length(c) == 0
  @test c.maxsize == 1 * sizeof(eltype(c))
end

@testset "aliasing" begin
  x = CuArray([1,2])
  y = view(x, 2:2)
  @test Base.mightalias(x, x)
  @test Base.mightalias(x, y)
  z = view(x, 1:1)
  @test Base.mightalias(x, z)
  @test !Base.mightalias(y, z)

  a = copy(y)::typeof(x)
  @test !Base.mightalias(x, a)
  a .= 3
  @test Array(y) == [2]

  b = Base.unaliascopy(y)::typeof(y)
  @test !Base.mightalias(x, b)
  b .= 3
  @test Array(y) == [2]
end

@testset "issue 919" begin
  # two-step mapreduce with wrapped CuArray as output
  @test vec(Array(sum!(view(CUDA.zeros(1,1,1), 1, :, :), CUDA.ones(1,4096)))) == [4096f0]
end

@testset "issue 1202" begin
  # test that deepcopying a struct with a CuArray field gets properly deepcopied
  a = (x = CUDA.zeros(2),)
  b = deepcopy(a)
  a.x .= -1
  @test b.x != a.x
end

@testset "isbits unions" begin
  # test that the selector bytes are preserved when up and downloading
  let a = [1, nothing, 3]
    b = CuArray(a)
    c = Array(b)
    @test a == c
  end

  # test that we can correctly read and write unions from device code
  let a = [1, nothing, 3]
    b = CuArray(a)
    c = similar(b, Bool)
    function kernel(x)
      i = threadIdx().x
      val = x[i]
      if val !== nothing
        x[i] = val + 1
      end
      return
    end
    @cuda threads=length(b) kernel(b)
    @test Array(b) == [2, nothing, 4]
  end

  # same for views
  let a = [0, nothing, 1, nothing, 3, nothing]
    b = CuArray(a)
    b = view(b, 3:5)
    c = Array(b)
    @test view(a, 3:5) == c
  end
  let a = [0, nothing, 1, nothing, 3, nothing]
    b = CuArray(a)
    b = view(b, 3:5)
    c = similar(b, Bool)
    function kernel(x)
      i = threadIdx().x
      val = x[i]
      if val !== nothing
        x[i] = val + 1
      end
      return
    end
    @cuda threads=length(b) kernel(b)
    @test Array(b) == [2, nothing, 4]
  end

  # test that we can create and use arrays with all singleton objects
  let a = [nothing, missing, missing, nothing]
    b = CuArray(a)
    errors = CuArray([0])
    function kernel()
      i = threadIdx().x
      if i == 1 || i == 4
        if b[i] !==  nothing
          errors[] += 1
        end
      else
        if b[i] !== missing
          errors[] += 1
        end
      end
      return
    end
    @cuda threads=length(b) kernel()
    @test Array(errors) == [0]
  end

  # arrays with union{union{...}, ...}
  let a = CuArray{Union{Union{Missing,Nothing},Int}}([0])
    function kernel(x)
      i = threadIdx().x
      val = x[i]
      if val !== nothing && val !== missing
        x[i] = val + 1
      end
      return
    end
    @cuda threads=length(a) kernel(a)
    @test Array(a) == [1]
  end

  # struct with unions are not isbits but are allocatedinline
  let
    @gensym typnam
    typ = @eval begin
      struct $typnam
        foo::Union{Int,Float32}
      end
      $typnam
    end
    a = CuArray{typ}(undef, 2)
    function kernel(x::AbstractArray{T}) where {T}
      i = threadIdx().x
      x[i] = if i == 1
        T(Int(i))
      else
        T(Float32(i))
      end
      return
    end
    @cuda threads=length(a) kernel(a)
    @test Array(a) == [typ(1), typ(2f0)]
  end
end

@testset "large map reduce" begin
  dev = device()

  big_size = CUDA.big_mapreduce_threshold(dev) + 5
  a = rand(Float32, big_size, 31)
  c = CuArray(a)

  expected = minimum(a, dims=2)
  actual = minimum(c, dims=2)
  @test expected == Array(actual)

  expected = findmax(a, dims=2)
  actual = findmax(c, dims=2)
  @test expected == map(Array, actual)

  expected = sum(a, dims=2)
  actual = sum(c, dims=2)
  @test expected == Array(actual)

  a = rand(Int, big_size, 31)
  c = CuArray(a)

  expected = minimum(a, dims=2)
  actual = minimum(c, dims=2)
  @test expected == Array(actual)

  expected = findmax(a, dims=2)
  actual = findmax(c, dims=2)
  @test expected == map(Array, actual)

  expected = sum(a, dims=2)
  actual = sum(c, dims=2)
  @test expected == Array(actual)

  @testset "#1169" begin
    function test_cuda_sum!(Nx, Ny, Nz)
        A = randn(Nx, Ny, Nz)
        R = zeros(1, Ny, Nz)
        dA = CuArray(A)
        dR = CuArray(R)
        sum!(dR, dA)
        sum!(R, A)
        R ≈ Array(dR)
    end

    @test test_cuda_sum!(32, 32, 32)
    @test test_cuda_sum!(256, 256, 256)
    @test test_cuda_sum!(512, 512, 512)
    @test test_cuda_sum!(85, 1320, 100)
  end
end

@testset "unified memory" begin
  dev = device()

  let
    a = CuVector{Int,CUDA.DeviceMemory}(undef, 1)
    @test is_device(a)
    @test !is_host(a)
    @test !is_unified(a)
    @test !is_managed(pointer(a))
  end

  let
    a = CuVector{Int,CUDA.UnifiedMemory}(undef, 1)
    @test !is_device(a)
    @test is_unified(a)
    @test !is_host(a)
    @test is_managed(pointer(a))
    a .= 0
    @test Array(a) == [0]

    if length(devices()) > 1
      other_devs = filter(!isequal(dev), collect(devices()))
      device!(first(other_devs)) do
        a .+= 1
        @test Array(a) == [1]
      end
      @test Array(a) == [1]
    end
  end

  let
    for B = [CUDA.DeviceMemory, CUDA.UnifiedMemory]
      a = CuVector{Float32,B}(rand(Float32, 1))
      @test !xor(B == CUDA.UnifiedMemory, is_unified(a))

      # check that buffer types are preserved
      let b = similar(a)
        @test eltype(b) == eltype(a)
        @test !xor(B == CUDA.UnifiedMemory, is_unified(b))
      end
      let b = CuArray(a)
        @test eltype(b) == eltype(a)
        @test !xor(B == CUDA.UnifiedMemory, is_unified(b))
      end
      let b = CuArray{Float64}(a)
        @test eltype(b) == Float64
        @test !xor(B == CUDA.UnifiedMemory, is_unified(b))
      end

      # change buffer type
      let b = CuVector{Float32,CUDA.DeviceMemory}(a)
        @test eltype(b) == eltype(a)
        @test !is_unified(b)
      end
      let b = CuVector{Float32,CUDA.UnifiedMemory}(a)
        @test eltype(b) == eltype(a)
        @test is_unified(b)
      end

      # change type and buffer type
      let b = CuVector{Float64,CUDA.DeviceMemory}(a)
        @test eltype(b) == Float64
        @test !is_unified(b)
      end
      let b = CuVector{Float64,CUDA.UnifiedMemory}(a)
        @test eltype(b) == Float64
        @test is_unified(b)
      end
    end

    # cu: supports unified keyword
    let a = cu(rand(Float64, 1); device=true)
      @test !is_unified(a)
      @test eltype(a) == Float32
    end
    let a = cu(rand(Float64, 1); unified=true)
      @test is_unified(a)
      @test eltype(a) == Float32
    end
  end
end

@testset "issue: invalid handling of device pointers" begin
  # failed when DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM == 0
  cpu = rand(2,2)
  buf = CUDA.register(CUDA.HostMemory, pointer(cpu), sizeof(cpu), CUDA.MEMHOSTREGISTER_DEVICEMAP)
  gpu_ptr = convert(CuPtr{eltype(cpu)}, buf)
  gpu = unsafe_wrap(CuArray, gpu_ptr, size(cpu))
  @test Array(gpu) == cpu
end

if length(devices()) > 1
@testset "multigpu" begin
  dev = device()
  other_devs = filter(!isequal(dev), collect(devices()))
  other_dev = first(other_devs)

  @testset "issue 1176" begin
    A = [1,2,3]
    dA = CuArray(A)
    synchronize()
    B = fetch(@async begin
        device!(other_dev)
        Array(dA)
    end)
    @test A == B
  end

  @testset "issue 1263" begin
    function unified_cuarray(::Type{T}, dims::NTuple{N}) where {T, N}
        buf = CUDA.alloc(CUDA.UnifiedMemory, prod(dims) * sizeof(T))
        array = unsafe_wrap(CuArray{T, N}, convert(CuPtr{T}, buf), dims)
        finalizer(_ -> CUDA.free(buf), array)
        return array
    end

    A = rand(5, 5)
    B = rand(5, 5)

    # should be able to copy to/from unified memory regardless of the context
    device!(dev)
    dA = CuArray(A)
    device!(other_dev)
    dB = unified_cuarray(eltype(A), size(A))
    device!(dev)
    copyto!(dB, dA)
    @test Array(dB) == A

    device!(other_dev)
    copyto!(dB, B)
    synchronize()
    device!(dev)
    copyto!(dA, dB)
    @test Array(dA) == B

    function host_cuarray(::Type{T}, dims::NTuple{N}) where {T, N}
        buf = CUDA.alloc(CUDA.UnifiedMemory, prod(dims) * sizeof(T))
        array = unsafe_wrap(CuArray{T, N}, convert(CuPtr{T}, buf), dims)
        finalizer(_ -> CUDA.free(buf), array)
        return array
    end

    # same for host memory
    device!(dev)
    dA = CuArray(A)
    device!(other_dev)
    dB = host_cuarray(eltype(A), size(A))
    device!(dev)
    copyto!(dB, dA)
    @test Array(dB) == A

    device!(other_dev)
    copyto!(dB, B)
    synchronize()
    device!(dev)
    copyto!(dA, dB)
    @test Array(dA) == B
  end

  @testset "issue 1136: copies between devices" begin
    device!(dev)
    data = rand(5, 5)
    a = CuArray(data)

    device!(other_dev)
    b = similar(a)
    @test device(b) == other_dev
    copyto!(b, a)

    synchronize()
    @test Array(a) == Array(b) == data

    device!(dev)
    @test Array(a) == Array(b) == data

    # now do the same, but with the other context active when copying
    device!(dev)
    data = rand(5, 5)
    a = CuArray(data)

    copyto!(b, a)

    synchronize()
    @test Array(a) == Array(b) == data

    device!(other_dev)
    @test Array(a) == Array(b) == data
  end
end
end

@testset "inplaceability" begin
  a = CUDA.rand(10, 10)
  @test is_inplaceable_destination(a)
  a′ = copy(a)
  b = CUDA.rand(10, 10)
  c = add!!(a, b)
  @test c == a′ + b
  @test c === a
end

@testset "issue 2595" begin
  # mixed-type reductions resulted in a deadlock because of union splitting over shfl
  a = CUDA.zeros(Float32, 1)
  b = CUDA.ones(Float64, 2)
  sum!(a, b)
  @test Array(a) == [2f0]
end
