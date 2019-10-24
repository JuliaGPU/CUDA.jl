using ForwardDiff: Dual
using LinearAlgebra
using Adapt: adapt

import CUDAdrv
import CUDAdrv: CuPtr, CU_NULL

@testset "GPUArrays test suite" begin
  GPUArrays.test(CuArray)
end

@testset "Memory" begin
  CuArrays.alloc(0)

  @test (CuArrays.@allocated CuArray{Int32}(undef,1)) == 4

  ret, out = @grab_output CuArrays.@time CuArray{Int32}(undef, 1)
  @test isa(ret, CuArray{Int32})
  @test occursin("1 GPU allocation: 4 bytes", out)

  ret, out = @grab_output CuArrays.@time Base.unsafe_wrap(CuArray, CuPtr{Int32}(12345678), (2, 3))
  @test isa(ret, CuArray{Int32})
  @test !occursin("GPU allocation", out)
end

@testset "Array" begin
  xs = CuArray{Int}(undef, 2, 3)
  @test collect(CuArray([1 2; 3 4])) == [1 2; 3 4]
  @test collect(cu[1, 2, 3]) == [1, 2, 3]
  @test collect(cu([1, 2, 3])) == [1, 2, 3]
  @test testf(vec, rand(5,3))
  @test cu(1:3) === 1:3
  @test Base.elsize(xs) == sizeof(Int)
  @test CuArray{Int, 2}(xs) === xs

  @test_throws ArgumentError Base.unsafe_convert(Ptr{Int}, xs)
  @test_throws ArgumentError Base.unsafe_convert(Ptr{Float32}, xs)

  # Check that allowscalar works
  @test_throws ErrorException xs[1]
  @test_throws ErrorException xs[1] = 1

  # unsafe_wrap
  @test Base.unsafe_wrap(CuArray, CU_NULL, 1; own=false).own == false
  @test Base.unsafe_wrap(CuArray, CU_NULL, 2)            == CuArray{Nothing,1}(CU_NULL, (2,), false)
  @test Base.unsafe_wrap(CuArray{Nothing}, CU_NULL, 2)   == CuArray{Nothing,1}(CU_NULL, (2,), false)
  @test Base.unsafe_wrap(CuArray{Nothing,1}, CU_NULL, 2) == CuArray{Nothing,1}(CU_NULL, (2,), false)
  @test Base.unsafe_wrap(CuArray, CU_NULL, (1,2))            == CuArray{Nothing,2}(CU_NULL, (1,2), false)
  @test Base.unsafe_wrap(CuArray{Nothing}, CU_NULL, (1,2))   == CuArray{Nothing,2}(CU_NULL, (1,2), false)
  @test Base.unsafe_wrap(CuArray{Nothing,2}, CU_NULL, (1,2)) == CuArray{Nothing,2}(CU_NULL, (1,2), false)

  @test collect(CuArrays.zeros(2, 2)) == zeros(Float32, 2, 2)
  @test collect(CuArrays.ones(2, 2)) == ones(Float32, 2, 2)

  @test collect(CuArrays.fill(0, 2, 2)) == zeros(Float32, 2, 2)
  @test collect(CuArrays.fill(1, 2, 2)) == ones(Float32, 2, 2)
end

@testset "Adapt" begin
  A = rand(Float32, 3, 3)
  dA = CuArray(A)
  @test adapt(Array, dA) ≈ A
  @test adapt(CuArray, A) ≈ dA
end

@testset "Broadcast" begin
  @test testf((x)       -> fill!(x, 1),  rand(3,3))
  @test testf((x, y)    -> map(+, x, y), rand(2, 3), rand(2, 3))
  @test testf((x)       -> sin.(x),      rand(2, 3))
  @test testf((x)       -> log.(x) .+ 1, rand(2, 3))
  @test testf((x)       -> 2x,           rand(2, 3))
  @test testf((x)       -> x .^ 0,      rand(2, 3))
  @test testf((x)       -> x .^ 1,      rand(2, 3))
  @test testf((x)       -> x .^ 2,      rand(2, 3))
  @test testf((x)       -> x .^ 3,      rand(2, 3))
  @test testf((x)       -> x .^ 5,      rand(2, 3))
  @test testf((x)       -> (z = Int32(5); x .^ z),      rand(2, 3))
  @test testf((x)       -> (z = Float64(π); x .^ z),      rand(2, 3))
  @test testf((x)       -> (z = Float32(π); x .^ z),      rand(Float32, 2, 3))
  @test testf((x, y)    -> x .+ y,       rand(2, 3), rand(1, 3))
  @test testf((z, x, y) -> z .= x .+ y,  rand(2, 3), rand(2, 3), rand(2))
  @test (CuArray{Ptr{Cvoid}}(undef, 1) .= C_NULL) == CuArray([C_NULL])
  @test CuArray([1,2,3]) .+ CuArray([1.0,2.0,3.0]) == CuArray([2,4,6])

  @eval struct Whatever{T}
      x::Int
  end
  @test Array(Whatever{Int}.(CuArray([1]))) == Whatever{Int}.([1])
end

@testset "Cufunc" begin
  gelu1(x) = oftype(x, 0.5) * x * (1 + tanh(oftype(x, √(2/π))*(x + oftype(x, 0.044715) * x^3)))
  sig(x) = one(x) / (one(x) + exp(-x))
  f(x) = gelu1(log(x)) * sig(x) * tanh(x)
  g(x) = x^7 - 2 * x^f(x^2) + 3


  CuArrays.@cufunc gelu1(x) = oftype(x, 0.5) * x * (1 + tanh(oftype(x, √(2/π))*(x + oftype(x, 0.044715) * x^3)))
  CuArrays.@cufunc sig(x) = one(x) / (one(x) + exp(-x))
  CuArrays.@cufunc f(x) = gelu1(log(x)) * sig(x) * tanh(x)
  CuArrays.@cufunc g(x) = x^7 - 2 * x^f(x^2) + 3

  @test :gelu1 ∈ CuArrays.cufuncs()
  @test :sig ∈ CuArrays.cufuncs()
  @test :f ∈ CuArrays.cufuncs()
  @test :g ∈ CuArrays.cufuncs()
  @test testf((x)  -> gelu1.(x), rand(3,3))
  @test testf((x)  -> sig.(x),  rand(3,3))
  @test testf((x)  -> f.(x),    rand(3,3))
  @test testf((x)  -> g.(x),    rand(3,3))
end

# https://github.com/JuliaGPU/CUDAnative.jl/issues/223
@testset "Ref Broadcast" begin
  foobar(idx, A) = A[idx]
  @test CuArray([42]) == foobar.(CuArray([1]), Base.RefValue(CuArray([42])))
end

@testset "Broadcast Fix" begin
  @test testf(x -> log.(x), rand(3,3))
  @test testf((x,xs) -> log.(x.+xs), Ref(1), rand(3,3))

  if CuArrays.libcudnn !== nothing
    using NNlib

    @test testf(x -> logσ.(x), rand(5))

    f(x) = logσ.(x)
    ds = Dual.(rand(5),1)
    @test f(ds) ≈ collect(f(CuArray(ds)))
  end
end

@testset "Reduce" begin
  @test testf(x -> sum(x, dims=1), rand(2, 3))
  @test testf(x -> sum(x, dims=2), rand(2, 3))
  @test testf(x -> sum(x -> x^2, x, dims=1), rand(2, 3))
  @test testf(x -> prod(x, dims=2), rand(2, 3))

  @test testf(x -> sum(x), rand(2, 3))
  @test testf(x -> prod(x), rand(2, 3))

  @test testf(x -> minimum(x), rand(2, 3))
  @test testf(x -> minimum(x, dims=2), rand(2, 3))
  @test testf(x -> minimum(x, dims=(2, 3)), rand(2, 3, 4))

  @test testf(x -> maximum(x), rand(2, 3))
  @test testf(x -> maximum(x, dims=2), rand(2, 3))
  @test testf(x -> maximum(x, dims=(2, 3)), rand(2, 3, 4))

  myreducer(x1, x2) = x1 + x2 # bypass optimisations for sum()
  @test testf(x -> reduce(myreducer, x, dims=(2, 3), init=0.0), rand(2, 3, 4))
  @test testf(x -> reduce(myreducer, x, init=0.0), rand(2, 3))
  @test testf(x -> reduce(myreducer, x, dims=2, init=42.0), rand(2, 3))

  ex = ErrorException("Please supply a neutral element for &. E.g: mapreduce(f, &, A; init = 1)")
  @test_throws ex mapreduce(t -> t > 0.5, &, cu(rand(2, 3)))
  @test testf(x -> mapreduce(t -> t > 0.5, &, x, init=true), rand(2, 3))

  ex = UndefKeywordError(:init)
  cub = map(t -> t > 0.5, cu(rand(2, 3)))
  @test_throws ex reduce(|, cub)
  @test testf(x -> reduce(|, x, init=false), map(t -> t > 0.5, cu(rand(2, 3))))
end

@testset "0D" begin
  x = CuArray{Float64}(undef)
  x .= 1
  @test collect(x)[] == 1
  if VERSION >= v"1.3-"
    # broken test that throws
    # https://github.com/JuliaGPU/GPUArrays.jl/issues/204
    @test_throws CUDAnative.InvalidIRError x /= 2
  else
    x /= 2
    @test collect(x)[] == 0.5
  end
end

@testset "SubArray" begin
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

  let x = cu(rand(Float32, 5, 4, 3))
    @test_throws BoundsError view(x, :, :, 1:10)

    # Contiguous views should return new CuArray
    @test typeof(view(x, :, 1, 2)) == CuVector{Float32}
    @test typeof(view(x, 1:4, 1, 2)) == CuVector{Float32}
    @test typeof(view(x, :, 1:4, 3)) == CuMatrix{Float32}
    @test typeof(view(x, :, :, 1)) == CuMatrix{Float32}
    @test typeof(view(x, :, :, :)) == CuArray{Float32,3}
    @test typeof(view(x, :)) == CuVector{Float32}
    @test typeof(view(x, 1:3)) == CuVector{Float32}

    # Non-contiguous views should fall back to base's SubArray
    @test typeof(view(x, 1:3, 1:3, 3)) <: SubArray
    @test typeof(view(x, 1, :, 3)) <: SubArray
    @test typeof(view(x, 1, 1:4, 3)) <: SubArray
    @test typeof(view(x, :, 1, 1:3)) <: SubArray
    @test typeof(view(x, :, 1:2:4, 1)) <: SubArray
    @test typeof(view(x, 1:2:5, 1, 1)) <: SubArray
  end

  # non-contiguous copyto!
  let x = CuArrays.rand(4, 4)
    y = view(x, 2:3, 2:3)

    # to gpu
    gpu = CuArray{eltype(y)}(undef, size(y))
    copyto!(gpu, y)
    @test Array(gpu) == Array(y)

    # to cpu
    cpu = Array{eltype(y)}(undef, size(y))
    copyto!(cpu, y)
    @test cpu == Array(y)
  end

  # bug in parentindices conversion
  let x = CuArray{Int}(undef, 1, 1)
    x[1,:] .= 42
    @test Array(x)[1,1] == 42
  end

  # bug in copyto!
  ## needless N type parameter
  @test testf((x,y)->copyto!(y, selectdim(x, 2, 1)), ones(2,2,2), zeros(2,2))
  ## inability to copyto! smaller destination
  @test testf((x,y)->copyto!(y, selectdim(x, 2, 1)), ones(2,2,2), zeros(3,3))
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
end

@testset "$f! with diagonal $d" for (f, f!) in ((triu, triu!), (tril, tril!)),
                                          d in -2:2
  A = randn(10, 10)
  @test f(A, d) == Array(f!(CuArray(A), d))
end

@testset "Utilities" begin
  t = Base.@elapsed ret = CuArrays.@sync begin
    # TODO: do something that takes a while on the GPU
    #       (need to wrap clock64 in CUDAnative for that)
    42
  end
  @test t >= 0
  @test ret == 42
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
end

@testset "generic fallbacks" begin
    a = rand(Int8, 3, 3)
    b = rand(Int8, 3, 3)
    d_a = CuArray{Int8}(a)
    d_b = CuArray{Int8}(b)
    d_c = d_a*d_b
    @test collect(d_c) == a*b
    a = rand(Complex{Int8}, 3, 3)
    b = rand(Complex{Int8}, 3, 3)
    d_a = CuArray{Complex{Int8}}(a)
    d_b = CuArray{Complex{Int8}}(b)
    d_c = d_a'*d_b
    @test collect(d_c) == a'*b
    d_c = d_a*d_b'
    @test collect(d_c) == a*b'
    d_c = d_a'*d_b'
    @test collect(d_c) == a'*b'
    d_c = transpose(d_a)*d_b'
    @test collect(d_c) == transpose(a)*b'
    d_c = d_a'*transpose(d_b)
    @test collect(d_c) == a'*transpose(b)
    d_c = transpose(d_a)*d_b
    @test collect(d_c) == transpose(a)*b
    d_c = d_a*transpose(d_b)
    @test collect(d_c) == a*transpose(b)
    d_c = transpose(d_a)*transpose(d_b)
    @test collect(d_c) == transpose(a)*transpose(b)
    d_c = rmul!(copy(d_a), Complex{Int8}(2, 2))
    @test collect(d_c) == a*Complex{Int8}(2, 2)
    d_c = lmul!(Complex{Int8}(2, 2), copy(d_a))
    @test collect(d_c) == Complex{Int8}(2, 2)*a
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
end

@testset "permutedims" begin
    @test testf(x->permutedims(x, [1, 2]), rand(4, 4))

    inds = rand(1:100, 150, 150)
    @test testf(x->permutedims(view(x, inds, :), (3, 2, 1)), rand(100, 100))
end

@testset "findall" begin
    @test testf(x->findall(x), rand(Bool, 100))
    @test testf(x->findall(y->y>0.5, x), rand(100))
end
