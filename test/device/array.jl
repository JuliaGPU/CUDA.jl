@testset "constructors" begin
    # inner constructors
    let
        dp = reinterpret(Core.LLVMPtr{Int,AS.Generic}, C_NULL)
        CuDeviceArray{Int,1,AS.Generic}((1,), dp)
    end

    # outer constructors
    T = Float32
    for (I1,I2) in [(Int, Int), (Int32, Int32), (Int32, Int64)]
        a = I1(1)
        b = I2(2)

        dp = reinterpret(CUDA.LLVMPtr{T,AS.Generic}, C_NULL)

        # not parameterized
        CuDeviceArray(b, dp)
        CuDeviceVector(b, dp)
        CuDeviceArray((b,), dp)
        CuDeviceVector((b,), dp)
        CuDeviceArray((b,a), dp)
        CuDeviceMatrix((b,a), dp)

        # partially parameterized
        CuDeviceArray{T}(b, dp)
        CuDeviceVector{T}(b, dp)
        CuDeviceArray{T}((b,), dp)
        CuDeviceVector{T}((b,), dp)
        CuDeviceArray{T}((a,b), dp)
        CuDeviceMatrix{T}((a,b), dp)
        CuDeviceArray{T,1}(b, dp)
        CuDeviceArray{T,1}((b,), dp)
        @test_throws MethodError CuDeviceArray{T,1}((a,b), dp)
        @test_throws MethodError CuDeviceArray{T,2}(b, dp)
        @test_throws MethodError CuDeviceArray{T,2}((b,), dp)
        CuDeviceArray{T,2}((a,b), dp)

        # fully parameterized
        CuDeviceArray{T,1,AS.Generic}(b, dp)
        CuDeviceArray{T,1,AS.Generic}((b,), dp)
        @test_throws MethodError CuDeviceArray{T,1,AS.Generic}((a,b), dp)
        @test_throws MethodError CuDeviceArray{T,1,AS.Shared}((a,b), dp)
        @test_throws MethodError CuDeviceArray{T,2,AS.Generic}(b, dp)
        @test_throws MethodError CuDeviceArray{T,2,AS.Generic}((b,), dp)
        CuDeviceArray{T,2,AS.Generic}((a,b), dp)

        # type aliases
        CuDeviceVector{T}(b, dp)
        CuDeviceMatrix{T}((a,b), dp)
    end
end

@testset "basics" begin     # argument passing, get and setindex, length
    dims = (16, 16)
    len = prod(dims)

    function kernel(input::CuDeviceArray{Float32}, output::CuDeviceArray{Float32})
        i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x

        if i <= length(input)
            output[i] = Float64(input[i])   # force conversion upon setindex!
        end

        return
    end

    input = round.(rand(Float32, dims) * 100)

    input_dev = CuArray(input)
    output_dev = CuArray(input)

    @test cudaconvert(input_dev) isa CuDeviceArray
    @test_throws ErrorException cudaconvert(input_dev)[]

    @cuda threads=len kernel(input_dev, output_dev)
    output = Array(output_dev)
    @test input ≈ output
end

@testset "iteration" begin     # argument passing, get and setindex, length
    dims = (16, 16)
    function kernel(input::CuDeviceArray{T}, output::CuDeviceArray{T}) where {T}
        acc = zero(T)
        for elem in input
            acc += elem
        end
        output[1] = acc
        return
    end

    input = round.(rand(Float32, dims) * 100)

    input_dev = CuArray(input)
    output_dev = CuArray(Float32[0])

    @cuda kernel(input_dev, output_dev)
    output = Array(output_dev)
    @test sum(input) ≈ output[1]
end

@testset "bounds checking" begin
    @testset "#313" begin
        function kernel(dest)
            dest[1] = 1
            nothing
        end
        tt = Tuple{SubArray{Float64,2,CuDeviceArray{Float64,2,AS.Global},
                            Tuple{UnitRange{Int64},UnitRange{Int64}},false}}

        ir = sprint(io->CUDA.code_llvm(io, kernel, tt))
        @test !occursin("jl_invoke", ir)
        CUDA.code_ptx(devnull, kernel, tt)
    end
end

@testset "views" begin
    function kernel(array)
        i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x

        _sub = view(array, 2:length(array)-1)
        if i <= length(_sub)
            _sub[i] = i
        end

        return
    end

    array = zeros(Int64, 100)
    array_dev = CuArray(array)

    sub = view(array, 2:length(array)-1)
    for i in 1:length(sub)
        sub[i] = i
    end

    @cuda threads=100 kernel(array_dev)
    @test array == Array(array_dev)
end

@testset "reshape" begin
    function kernel(array)
        i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
        j = (blockIdx().y-1i32) * blockDim().y + threadIdx().y

        _array2d = reshape(array, 10, 10)
        _array2d[i,j] = i + (j-1)*size(_array2d,1)

        return
    end

    array = zeros(Int64, 100)
    array_dev = CuArray(array)

    array2d = reshape(array, 10, 10)
    for i in 1:size(array2d,1), j in 1:size(array2d,2)
        array2d[i,j] = i + (j-1)*size(array2d,1)
    end

    @cuda threads=(10, 10) kernel(array_dev)
    @test array == Array(array_dev)
end

@testset "non-Int index to unsafe_load" begin
    function kernel(a)
        a[UInt64(1)] = 1
        return
    end

    array = CUDA.zeros(1)
    @cuda kernel(array)
    @test Array(array) == [1]
end


function kernel_shmem_reinterpet_equal_size!(y)
  a = CuDynamicSharedArray(Float32, (blockDim().x,))
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
  a = CuDynamicSharedArray(UInt128, (blockDim().x,))
  i = threadIdx().x
  p = i + i * im
  q = i - i * im
  b = reinterpret(typeof(p), a)
  b[1 + 2 * (threadIdx().x - 1i32)] = p
  b[2 + 2 * (threadIdx().x - 1i32)] = q
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
  a = CuDynamicSharedArray(Float32, (4 * blockDim().x,))
  b = reinterpret(UInt128, a)
  a[1 + 4 * (threadIdx().x - 1i32)] = threadIdx().x
  a[2 + 4 * (threadIdx().x - 1i32)] = threadIdx().x * 2
  a[3 + 4 * (threadIdx().x - 1i32)] = threadIdx().x * 3
  a[4 + 4 * (threadIdx().x - 1i32)] = threadIdx().x * 4
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
