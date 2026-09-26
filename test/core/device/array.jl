@testset "constructors" begin
    T = Float32
    for (I1,I2) in [(Int, Int), (Int32, Int32), (Int32, Int64)]
        a = I1(1)
        b = I2(2)

        dp = reinterpret(Core.LLVMPtr{T,AS.Generic}, C_NULL)

        CuDeviceArray{T,1,AS.Generic}(dp, (b,))
        @test_throws MethodError CuDeviceArray{T,1,AS.Generic}(dp, (a,b))
        @test_throws MethodError CuDeviceArray{T,1,AS.Shared}(dp, (a,b))
        @test_throws MethodError CuDeviceArray{T,2,AS.Generic}(dp, (b,))
        CuDeviceArray{T,2,AS.Generic}(dp, (a,b))

        # type aliases
        CuDeviceVector{T,AS.Generic}(dp, (b,))
        CuDeviceMatrix{T,AS.Generic}(dp, (a,b))
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
    @test_throws ErrorException cudaconvert(input_dev)[1]
    @test_throws ErrorException cudaconvert(input_dev)[1, 1]

    @cuda threads=len kernel(input_dev, output_dev)
    output = Array(output_dev)
    @test input ≈ output
end

@testset "index type" begin
    a = CuArray{Float32}(undef, 2, 3)
    da = cudaconvert(a)
    @test da isa CuDeviceArray{Float32,2,AS.Global,Int32}
    # the index type doesn't change the array interface
    @test size(da) === (2, 3)
    @test length(da) === 6

    # arrays that are too large for 32-bit indices (only converted, never accessed)
    GC.@preserve a begin
        big = unsafe_wrap(CuArray, pointer(a), (2, 2^30))
        @test cudaconvert(big) isa CuDeviceArray{Float32,2,AS.Global,Int64}
        @test size(cudaconvert(big)) === (2, 2^30)

        # every dimension needs to fit too, not only the length
        empty = unsafe_wrap(CuArray, pointer(a), (0, 2^40))
        @test cudaconvert(empty) isa CuDeviceArray{Float32,2,AS.Global,Int64}
        @test size(cudaconvert(empty)) === (0, 2^40)

        # empty arrays whose dimensions fit, but whose partial products don't
        empty = unsafe_wrap(CuArray, pointer(a), (2^16, 2^16, 0))
        @test cudaconvert(empty) isa CuDeviceArray{Float32,3,AS.Global,Int32}
    end

    ptr = reinterpret(Core.LLVMPtr{Float32,AS.Global}, C_NULL)
    @test_throws ArgumentError CuDeviceArray{Float32,2,AS.Global,Int32}(ptr, (2, 2^30))
    @test_throws ArgumentError CuDeviceArray{Float32,2,AS.Global,Int32}(ptr, (0, 2^40))
    @test_throws ArgumentError CuDeviceArray{Float32,2,AS.Global,Int64}(ptr, (0, Int128(2)^64), 0)

    # shared memory checks dimensions that don't fit, even if the array is empty
    # (throwing on the device would break the context, so only check the code)
    @test @filecheck CUDA.code_llvm(Tuple{CuDeviceVector{Int,AS.Global,Int32},Int}) do out, n
        @check "throw_index_type_error"
        sh = CuDynamicSharedArray(UInt8, (0, n))
        @inbounds out[1] = size(sh, 2)
        return
    end

    # both index types compute the same thing
    function kernel(B, A)
        i = threadIdx().x
        j = blockIdx().x
        @inbounds B[i, j] = 2 * A[i, j] + A[CartesianIndex(i, j)] + A[i + (j - 1) * size(A, 1)]
        return
    end
    A = CUDA.rand(Float32, 7, 5)
    B = CUDA.zeros(Float32, 7, 5)
    @cuda threads=7 blocks=5 kernel(B, A)
    @test Array(B) ≈ 4 .* Array(A)
    GC.@preserve A B begin
        dA = Base.unsafe_convert(CuDeviceArray{Float32,2,AS.Global,Int64}, A)
        dB = Base.unsafe_convert(CuDeviceArray{Float32,2,AS.Global,Int64}, B)
        fill!(B, 0)
        @cuda threads=7 blocks=5 kernel(dB, dA)
        @test Array(B) ≈ 4 .* Array(A)

        # kernels compiled for one index type can be called with arrays of another
        k = @cuda launch=false kernel(B, A)
        fill!(B, 0)
        k(dB, dA; threads=7, blocks=5)
        @test Array(B) ≈ 4 .* Array(A)
    end

    # multidimensional indexing of arrays with 32-bit indices uses 32-bit arithmetic
    @test @filecheck CUDA.code_ptx(Tuple{CuDeviceArray{Float32,3,AS.Global,Int32},Int,Int,Int}) do A, i, j, k
        @check_not "mul.lo.s64"
        @check_not "mad.lo.s64"
        @inbounds A[i, j, k] = 1
        return
    end
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
    @testset "multidimensional indices" begin
        # every index is checked against its dimension, not only the linearized index.
        # these checks happen before the (device-only) memory access, so work on the host.
        A = cudaconvert(CuArray{Float32}(undef, 2, 3))
        @test_throws BoundsError A[3, 1]    # linearizes to 3, which is in bounds
        @test_throws BoundsError A[0, 2]    # linearizes to 2
        @test_throws BoundsError A[CartesianIndex(3, 1)]
        @test_throws BoundsError A[1, 0x4]
        @test_throws BoundsError A[1, 1, 2]
        @test_throws BoundsError (A[3, 1] = 1)
        @test_throws ErrorException A[2, 3]
        @test_throws ErrorException A[2, 3, 1]
        @test_throws ErrorException A[CartesianIndex(2, 3)]
    end

    @testset "#313" begin
        kernel = dest -> (dest[1] = 1; nothing)
        tt = Tuple{SubArray{Float64,2,CuDeviceArray{Float64,2,AS.Global,Int32},
                            Tuple{UnitRange{Int64},UnitRange{Int64}},false}}
        @test @filecheck CUDA.code_llvm(tt) do dest
            @check_not "jl_invoke"
            dest[1] = 1
            nothing
        end
        # also smoke-test that PTX codegen succeeds for this signature.
        CUDA.code_ptx(devnull, kernel, tt)
    end

    # test that we don't do needless bounds checking when the kernel already does it
    # (enabled by the fact that we store `len` next to `dims`)
    for N in 1:3, I in (Int32, Int64)
        @test @filecheck CUDA.code_llvm(Tuple{CuDeviceArray{Int,N,AS.Global,I}}) do A
            @check_not "boundserror"
            idx = threadIdx().x
            if idx <= length(A)
                # we did our own bounds checking, so no check should be left!
                A[idx] = 1
            end
            return
        end
    end
    for I in (Int32, Int64)
        @test @filecheck CUDA.code_llvm(Tuple{CuDeviceArray{Int,2,AS.Global,I}}) do A
            @check_not "boundserror"
            i = threadIdx().x
            j = blockIdx().x
            if i <= size(A, 1) && j <= size(A, 2)
                A[i, j] = 1
            end
            return
        end
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

@testset "reshape of view" begin
    function kernel(out, data, n)
        i = threadIdx().x
        if i <= n * n
            mat = reshape(@view(data[1:n*n]), (n, n))
            out[i] = mat[i]
        end
        return
    end

    n = 4
    data = CuArray(Float32.(1:n*n))
    out = CUDA.zeros(Float32, n * n)

    @cuda threads=n*n kernel(out, data, n)
    @test Array(out) == Float32.(1:n*n)
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
