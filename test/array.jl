using LinearAlgebra
import Adapt

@testset "constructors" begin
  xs = CuArray{Int}(undef, 2, 3)
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
  @test Base.unsafe_wrap(CuArray, CU_NULL, 1; own=false).state == CUDA.ARRAY_UNMANAGED
  ## compare structs without using mapreduce
  structeq(a,b) = false
  structeq(a::T,b::T) where T = all(field->getfield(a,field) == getfield(b,field), fieldnames(T))
  @test structeq(Base.unsafe_wrap(CuArray, CU_NULL, 2),                CuArray{Nothing,1}(CU_NULL, (2,)))
  @test structeq(Base.unsafe_wrap(CuArray{Nothing}, CU_NULL, 2),       CuArray{Nothing,1}(CU_NULL, (2,)))
  @test structeq(Base.unsafe_wrap(CuArray{Nothing,1}, CU_NULL, 2),     CuArray{Nothing,1}(CU_NULL, (2,)))
  @test structeq(Base.unsafe_wrap(CuArray, CU_NULL, (1,2)),            CuArray{Nothing,2}(CU_NULL, (1,2)))
  @test structeq(Base.unsafe_wrap(CuArray{Nothing}, CU_NULL, (1,2)),   CuArray{Nothing,2}(CU_NULL, (1,2)))
  @test structeq(Base.unsafe_wrap(CuArray{Nothing,2}, CU_NULL, (1,2)), CuArray{Nothing,2}(CU_NULL, (1,2)))

  @test collect(CUDA.zeros(2, 2)) == zeros(Float32, 2, 2)
  @test collect(CUDA.ones(2, 2)) == ones(Float32, 2, 2)

  @test collect(CUDA.fill(0, 2, 2)) == zeros(Float32, 2, 2)
  @test collect(CUDA.fill(1, 2, 2)) == ones(Float32, 2, 2)
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


function kernel_shmem_reinterpet_equal_size!(y)
  a = @cuDynamicSharedMem(Float32, (blockDim().x,))
  b = reinterpret(UInt32, a)
  a[threadIdx().x] = threadIdx().x
  b[threadIdx().x] += 1
  y[threadIdx().x] = a[threadIdx().x]
  return
end

function shmem_reinterpet_equal_size()
  threads = 4
  y = CUDA.zeros(threads)
  shmem = sizeof(Float32) * threads
  @cuda(
    threads = threads,
    blocks = 1,
    shmem = shmem,
    kernel_shmem_reinterpet_equal_size!(y)
  )
  return y
end

@testset "reinterpret shmem: equal size" begin
  gpu = shmem_reinterpet_equal_size()
  a = zeros(Float32, length(gpu))
  b = reinterpret(UInt32, a)
  a .= 1:length(b)
  b .+= 1
  @test collect(gpu) == a
end

function kernel_shmem_reinterpet_smaller_size!(y)
  a = @cuDynamicSharedMem(UInt128, (blockDim().x,))
  i32 = Int32(threadIdx().x)
  p = i32 + i32 * im
  q = i32 - i32 * im
  b = reinterpret(typeof(p), a)
  b[1 + 2 * (threadIdx().x - 1)] = p
  b[2 + 2 * (threadIdx().x - 1)] = q
  y[threadIdx().x] = a[threadIdx().x]
  return
end

function shmem_reinterpet_smaller_size()
  threads = 4
  y = CUDA.zeros(UInt128, threads)
  shmem = sizeof(UInt128) * threads
  @cuda(
    threads = threads,
    blocks = 1,
    shmem = shmem,
    kernel_shmem_reinterpet_smaller_size!(y)
  )
  return y
end

@testset "reinterpret shmem: smaller size" begin
  gpu = shmem_reinterpet_smaller_size()
  n = length(gpu)
  a = zeros(UInt128, n)
  p(i) = Int32(i) + Int32(i) * im
  q(i) = Int32(i) - Int32(i) * im
  b = reinterpret(typeof(p(0)), a)
  b[1:2:end] .= p.(1:n)
  b[2:2:end] .= q.(1:n)
  @test collect(gpu) == a
end

function kernel_shmem_reinterpet_larger_size!(y)
  a = @cuDynamicSharedMem(Float32, (4 * blockDim().x,))
  b = reinterpret(UInt128, a)
  a[1 + 4 * (threadIdx().x - 1)] = threadIdx().x
  a[2 + 4 * (threadIdx().x - 1)] = threadIdx().x * 2
  a[3 + 4 * (threadIdx().x - 1)] = threadIdx().x * 3
  a[4 + 4 * (threadIdx().x - 1)] = threadIdx().x * 4
  y[threadIdx().x] = b[threadIdx().x]
  return
end

function shmem_reinterpet_larger_size()
  threads = 4
  y = CUDA.zeros(UInt128, threads)
  shmem = sizeof(UInt128) * threads
  @cuda(
    threads = threads,
    blocks = 1,
    shmem = shmem,
    kernel_shmem_reinterpet_larger_size!(y)
  )
  return y
end

@testset "reinterpret shmem: larger size" begin
  gpu = shmem_reinterpet_larger_size()
  n = length(gpu)
  b = zeros(UInt128, n)
  a = reinterpret(Float32, b)
  a[1:4:end] .= 1:n
  a[2:4:end] .= (1:n) .* 2
  a[3:4:end] .= (1:n) .* 3
  a[4:4:end] .= (1:n) .* 4
  @test collect(gpu) == b
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
    @test testf(x->accumulate(+, x; dims=dims, init=100.), rand(Int, sizes))
  end

  # in place
  @test testf(x->(accumulate!(+, x, copy(x)); x), rand(2))

  # specialized
  @test testf(cumsum, rand(2))
  @test testf(cumprod, rand(2))
end

@testset "logical indexing" begin
  @test CuArray{Int}(undef, 2)[CuArray{Bool}(undef, 2)] isa CuArray
  @test CuArray{Int}(undef, 2, 2)[CuArray{Bool}(undef, 2, 2)] isa CuArray
  @test CuArray{Int}(undef, 2, 2, 2)[CuArray{Bool}(undef, 2, 2, 2)] isa CuArray

  @test CuArray{Int}(undef, 2)[Array{Bool}(undef, 2)] isa CuArray
  @test CuArray{Int}(undef, 2, 2)[Array{Bool}(undef, 2, 2)] isa CuArray
  @test CuArray{Int}(undef, 2, 2, 2)[Array{Bool}(undef, 2, 2, 2)] isa CuArray

  @test testf((x,y)->x[y], rand(2), rand(Bool, 2))
  @test testf((x,y)->x[y], rand(2, 2), rand(Bool, 2, 2))
  @test testf((x,y)->x[y], rand(2, 2, 2), rand(Bool, 2, 2, 2))

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
