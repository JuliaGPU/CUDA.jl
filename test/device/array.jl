@testset "constructors" begin
    # inner constructors
    let
        dp = reinterpret(Core.LLVMPtr{Int,AS.Generic}, C_NULL)
        CuDeviceArray{Int,1,AS.Generic}((1,), dp)
    end

    # outer constructors
    for I in [Int32,Int64]
        a = I(1)
        b = I(2)

        dp = reinterpret(CUDA.LLVMPtr{I,AS.Generic}, C_NULL)

        # not parameterized
        CuDeviceArray(b, dp)
        CuDeviceArray((b,), dp)
        CuDeviceArray((b,a), dp)

        # partially parameterized
        CuDeviceArray{I}(b, dp)
        CuDeviceArray{I}((b,), dp)
        CuDeviceArray{I}((a,b), dp)
        CuDeviceArray{I,1}(b, dp)
        CuDeviceArray{I,1}((b,), dp)
        @test_throws MethodError CuDeviceArray{I,1}((a,b), dp)
        @test_throws MethodError CuDeviceArray{I,2}(b, dp)
        @test_throws MethodError CuDeviceArray{I,2}((b,), dp)
        CuDeviceArray{I,2}((a,b), dp)

        # fully parameterized
        CuDeviceArray{I,1,AS.Generic}(b, dp)
        CuDeviceArray{I,1,AS.Generic}((b,), dp)
        @test_throws MethodError CuDeviceArray{I,1,AS.Generic}((a,b), dp)
        @test_throws MethodError CuDeviceArray{I,1,AS.Shared}((a,b), dp)
        @test_throws MethodError CuDeviceArray{I,2,AS.Generic}(b, dp)
        @test_throws MethodError CuDeviceArray{I,2,AS.Generic}((b,), dp)
        CuDeviceArray{I,2,AS.Generic}((a,b), dp)

        # type aliases
        CuDeviceVector{I}(b, dp)
        CuDeviceMatrix{I}((a,b), dp)
    end
end

@testset "basics" begin     # argument passing, get and setindex, length
    dims = (16, 16)
    len = prod(dims)

    function kernel(input::CuDeviceArray{Float32}, output::CuDeviceArray{Float32})
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x

        if i <= length(input)
            output[i] = Float64(input[i])   # force conversion upon setindex!
        end

        return
    end

    input = round.(rand(Float32, dims) * 100)

    input_dev = CuArray(input)
    output_dev = CuArray(input)

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
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x

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

@testset "non-Int index to unsafe_load" begin
    function load_index(a)
        return a[UInt64(1)]
    end

    a = [1]
    p = pointer(a)
    dp = reinterpret(Core.LLVMPtr{eltype(p), AS.Generic}, p)
    da = CUDA.CuDeviceArray(1, dp)
    load_index(da)
end
