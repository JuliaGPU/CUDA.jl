using LinearAlgebra
import Adapt

@testset "constructors" begin
  xs = CuArray{Int}(undef, 2, 3)
  @test device(xs) == device()
  @test context(xs) == context()
  @test collect(CuArray([1 2; 3 4])) == [1 2; 3 4]
  @test collect(cu[1, 2, 3]) == [1, 2, 3]
  @test collect(cu([1, 2, 3])) == [1, 2, 3]
  @test testf(vec, rand(5,3))
  @test cu(1:3) === 1:3
  @test Base.elsize(xs) == sizeof(Int)
  @test CuArray{Int, 2}(xs) === xs

  # test aggressive conversion to Float32, but only for floats, and only with `cu`
  @test cu([1]) isa AbstractArray{Int}
  @test cu(Float64[1]) isa AbstractArray{Float32}
  @test cu(ComplexF64[1+1im]) isa AbstractArray{ComplexF32}
  @test Adapt.adapt(CuArray, Float64[1]) isa AbstractArray{Float64}
  @test Adapt.adapt(CuArray, ComplexF64[1]) isa AbstractArray{ComplexF64}
  @test Adapt.adapt(CuArray{Float16}, Float64[1]) isa AbstractArray{Float16}

  @test_throws ArgumentError Base.unsafe_convert(Ptr{Int}, xs)
  @test_throws ArgumentError Base.unsafe_convert(Ptr{Float32}, xs)

  # unsafe_wrap
  let
    data = CuArray{Int}(undef, 2)
    ptr = pointer(data)
    B = Mem.DeviceBuffer

    @test unsafe_wrap(CuArray, ptr, 1; own=false).storage.refcount[] == -1

    ## compare the fields we care about: the buffer, size, offset, and context
    function test_eq(a, b)
      @test eltype(a) == eltype(b)
      @test ndims(a) == ndims(b)
      @test a.storage.buffer.ptr == b.storage.buffer.ptr
      @test a.storage.buffer.ctx == b.storage.buffer.ctx
      @test a.maxsize == b.maxsize
      @test a.offset == b.offset
      @test a.dims == b.dims
    end

    test_eq(unsafe_wrap(CuArray, ptr, 2),                CuArray{Int,1}(data.storage, (2,)))
    test_eq(unsafe_wrap(CuArray{Int}, ptr, 2),           CuArray{Int,1}(data.storage, (2,)))
    test_eq(unsafe_wrap(CuArray{Int,1}, ptr, 2),         CuArray{Int,1}(data.storage, (2,)))
    test_eq(unsafe_wrap(CuArray{Int,1,B}, ptr, 2),       CuArray{Int,1}(data.storage, (2,)))
    test_eq(unsafe_wrap(CuArray, ptr, (1,2)),            CuArray{Int,2}(data.storage, (1,2)))
    test_eq(unsafe_wrap(CuArray{Int}, ptr, (1,2)),       CuArray{Int,2}(data.storage, (1,2)))
    test_eq(unsafe_wrap(CuArray{Int,2}, ptr, (1,2)),     CuArray{Int,2}(data.storage, (1,2)))
    test_eq(unsafe_wrap(CuArray{Int,2,B}, ptr, (1,2)),   CuArray{Int,2}(data.storage, (1,2)))

    @test_throws ErrorException unsafe_wrap(CuArray{Int,1,Mem.HostBuffer}, ptr, 2)
    @test_throws ErrorException unsafe_wrap(CuArray{Int,2,Mem.HostBuffer}, ptr, (1,2))
  end
  let buf = Mem.alloc(Mem.Host, sizeof(Int), Mem.HOSTALLOC_DEVICEMAP)
    gpu_ptr = convert(CuPtr{Int}, buf)
    gpu_arr = unsafe_wrap(CuArray, gpu_ptr, 1)
    gpu_arr .= 42

    synchronize()

    cpu_ptr = convert(Ptr{Int}, buf)
    cpu_arr = unsafe_wrap(Array, cpu_ptr, 1)
    @test cpu_arr == [42]
  end

  @test collect(CUDA.zeros(2, 2)) == zeros(Float32, 2, 2)
  @test collect(CUDA.ones(2, 2)) == ones(Float32, 2, 2)

  @test collect(CUDA.fill(0, 2, 2)) == zeros(Float32, 2, 2)
  @test collect(CUDA.fill(1, 2, 2)) == ones(Float32, 2, 2)

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

@testset "adapt" begin
  A = rand(Float32, 3, 3)
  dA = CuArray(A)
  @test Adapt.adapt(Array, dA) == A
  @test Adapt.adapt(CuArray, A) isa CuArray
  @test Array(Adapt.adapt(CuArray, A)) == A
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
      @test testf(view, a, i)
      @test testf(view, a, view(i, 2:2))
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
    @test testf((x,y)->accumulate(+, x; init=y), rand(n), rand())
  end

  # multidimensional
  for (sizes, dims) in ((2,) => 2,
                        (3,4,5) => 2,
                        (1, 70, 50, 20) => 3)
    @test testf(x->accumulate(+, x; dims=dims), rand(Int, sizes))
  end

  # using initializer
  for (sizes, dims) in ((2,) => 2,
                        (3,4,5) => 2,
                        (1, 70, 50, 20) => 3)
    @test testf((x,y)->accumulate(+, x; dims=dims, init=y), rand(Int, sizes), rand(Int))
  end

  # in place
  @test testf(x->(accumulate!(+, x, copy(x)); x), rand(2))

  # specialized
  @test testf(cumsum, rand(2))
  @test testf(cumprod, rand(2))
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
    @test testf(x->findall(x), rand(Bool, 100))
    @test testf(x->findall(y->y>0.5, x), rand(100))

    # ND
    let x = rand(Bool, 10, 10)
      @test findall(x) == Array(findall(CuArray(x)))
    end
    let x = rand(10, 10)
      @test findall(y->y>0.5, x) == Array(findall(y->y>0.5, CuArray(x)))
    end
end

@testset "findfirst" begin
    # 1D
    @test testf(x->findfirst(x), rand(Bool, 100))
    @test testf(x->findfirst(y->y>0.5, x), rand(100))
    let x = fill(false, 10)
      @test findfirst(x) == findfirst(CuArray(x))
    end

    # ND
    let x = rand(Bool, 10, 10)
      @test findfirst(x) == findfirst(CuArray(x))
    end
    let x = rand(10, 10)
      @test findfirst(y->y>0.5, x) == findfirst(y->y>0.5, CuArray(x))
    end
end

@testset "findmax & findmin" begin
  let x = rand(Float32, 100)
      @test findmax(x) == findmax(CuArray(x))
      @test findmax(x; dims=1) == Array.(findmax(CuArray(x); dims=1))

      x[32] = x[33] = x[55] = x[66] = NaN32
      @test isequal(findmax(x), findmax(CuArray(x)))
      @test isequal(findmax(x; dims=1), Array.(findmax(CuArray(x); dims=1)))
  end
  let x = rand(Float32, 10, 10)
      @test findmax(x) == findmax(CuArray(x))
      @test findmax(x; dims=1) == Array.(findmax(CuArray(x); dims=1))
      @test findmax(x; dims=2) == Array.(findmax(CuArray(x); dims=2))

      x[rand(CartesianIndices((10, 10)), 10)] .= NaN
      @test isequal(findmax(x), findmax(CuArray(x)))
      @test isequal(findmax(x; dims=1), Array.(findmax(CuArray(x); dims=1)))
  end
  let x = rand(Float32, 10, 10, 10)
      @test findmax(x) == findmax(CuArray(x))
      @test findmax(x; dims=1) == Array.(findmax(CuArray(x); dims=1))
      @test findmax(x; dims=2) == Array.(findmax(CuArray(x); dims=2))
      @test findmax(x; dims=3) == Array.(findmax(CuArray(x); dims=3))

      x[rand(CartesianIndices((10, 10, 10)), 20)] .= NaN
      @test isequal(findmax(x), findmax(CuArray(x)))
      @test isequal(findmax(x; dims=1), Array.(findmax(CuArray(x); dims=1)))
      @test isequal(findmax(x; dims=2), Array.(findmax(CuArray(x); dims=2)))
      @test isequal(findmax(x; dims=3), Array.(findmax(CuArray(x); dims=3)))
  end

  let x = rand(Float32, 100)
      @test findmin(x) == findmin(CuArray(x))
      @test findmin(x; dims=1) == Array.(findmin(CuArray(x); dims=1))

      x[32] = x[33] = x[55] = x[66] = NaN32
      @test isequal(findmin(x), findmin(CuArray(x)))
      @test isequal(findmin(x; dims=1), Array.(findmin(CuArray(x); dims=1)))
  end
  let x = rand(Float32, 10, 10)
      @test findmin(x) == findmin(CuArray(x))
      @test findmin(x; dims=1) == Array.(findmin(CuArray(x); dims=1))
      @test findmin(x; dims=2) == Array.(findmin(CuArray(x); dims=2))

      x[rand(CartesianIndices((10, 10)), 10)] .= NaN
      @test isequal(findmin(x), findmin(CuArray(x)))
      @test isequal(findmin(x; dims=1), Array.(findmin(CuArray(x); dims=1)))
      @test isequal(findmin(x; dims=2), Array.(findmin(CuArray(x); dims=2)))
      @test isequal(findmin(x; dims=3), Array.(findmin(CuArray(x); dims=3)))
  end
  let x = rand(Float32, 10, 10, 10)
      @test findmin(x) == findmin(CuArray(x))
      @test findmin(x; dims=1) == Array.(findmin(CuArray(x); dims=1))
      @test findmin(x; dims=2) == Array.(findmin(CuArray(x); dims=2))
      @test findmin(x; dims=3) == Array.(findmin(CuArray(x); dims=3))

      x[rand(CartesianIndices((10, 10, 10)), 20)] .= NaN
      @test isequal(findmin(x), findmin(CuArray(x)))
      @test isequal(findmin(x; dims=1), Array.(findmin(CuArray(x); dims=1)))
      @test isequal(findmin(x; dims=2), Array.(findmin(CuArray(x); dims=2)))
      @test isequal(findmin(x; dims=3), Array.(findmin(CuArray(x); dims=3)))
  end
end

@testset "argmax & argmin" begin
    @test testf(argmax, rand(Int, 10))
    @test testf(argmax, -rand(Int, 10))

    @test testf(argmin, rand(Int, 10))
    @test testf(argmin, -rand(Int, 10))
end

@testset "issue #543" begin
  x = CUDA.rand(ComplexF32, 1)
  @test x isa CuArray{Complex{Float32}}

  y = exp.(x)
  @test y isa CuArray{Complex{Float32}}
end

@testset "resizing" begin
    a = CuArray([1,2,3])

    resize!(a, 3)
    @test length(a) == 3
    @test Array(a) == [1,2,3]

    resize!(a, 5)
    @test length(a) == 5
    @test Array(a)[1:3] == [1,2,3]

    resize!(a, 2)
    @test length(a) == 2
    @test Array(a)[1:2] == [1,2]

    GC.@preserve a begin
      b = unsafe_wrap(CuArray{Int}, pointer(a), 2)
      @test_throws ArgumentError resize!(b, 3)
    end

    b = CuArray{Int}(undef, 0)
    @test length(b) == 0
    resize!(b, 1)
    @test length(b) == 1
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
    a = CuVector{Int}(undef, 1)
    @test !is_unified(a)
    @test !is_managed(pointer(a))
  end

  let
    a = CuVector{Int,Mem.UnifiedBuffer}(undef, 1)
    @test is_unified(a)
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
    # default ctor: device memory
    let a = CUDA.rand(1)
      @test !is_unified(a)
      @test !is_managed(pointer(a))
    end

    for B = [Mem.DeviceBuffer, Mem.UnifiedBuffer]
      a = CuVector{Float32,B}(rand(Float32, 1))
      @test !xor(B == Mem.UnifiedBuffer, is_unified(a))

      # check that buffer types are preserved
      let b = similar(a)
        @test eltype(b) == eltype(a)
        @test !xor(B == Mem.UnifiedBuffer, is_unified(b))
      end
      let b = CuArray(a)
        @test eltype(b) == eltype(a)
        @test !xor(B == Mem.UnifiedBuffer, is_unified(b))
      end
      let b = CuArray{Float64}(a)
        @test eltype(b) == Float64
        @test !xor(B == Mem.UnifiedBuffer, is_unified(b))
      end

      # change buffer type
      let b = CuVector{Float32,Mem.DeviceBuffer}(a)
        @test eltype(b) == eltype(a)
        @test !is_unified(b)
      end
      let b = CuVector{Float32,Mem.UnifiedBuffer}(a)
        @test eltype(b) == eltype(a)
        @test is_unified(b)
      end

      # change type and buffer type
      let b = CuVector{Float64,Mem.DeviceBuffer}(a)
        @test eltype(b) == Float64
        @test !is_unified(b)
      end
      let b = CuVector{Float64,Mem.UnifiedBuffer}(a)
        @test eltype(b) == Float64
        @test is_unified(b)
      end
    end

    # cu: supports unified keyword
    let a = cu(rand(Float64, 1); unified=true)
      @test is_unified(a)
      @test eltype(a) == Float32
    end
    let a = cu(rand(Float64, 1))
      @test !is_unified(a)
      @test eltype(a) == Float32
    end
  end
end

@testset "issue: invalid handling of device pointers" begin
  # failed when DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM == 0
  cpu = rand(2,2)
  buf = Mem.register(Mem.Host, pointer(cpu), sizeof(cpu), Mem.HOSTREGISTER_DEVICEMAP)
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
        buf = Mem.alloc(Mem.Unified, prod(dims) * sizeof(T))
        array = unsafe_wrap(CuArray{T, N}, convert(CuPtr{T}, buf), dims)
        finalizer(_ -> Mem.free(buf), array)
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
        buf = Mem.alloc(Mem.Unified, prod(dims) * sizeof(T))
        array = unsafe_wrap(CuArray{T, N}, convert(CuPtr{T}, buf), dims)
        finalizer(_ -> Mem.free(buf), array)
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
