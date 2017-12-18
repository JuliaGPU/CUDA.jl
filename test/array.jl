@testset "device arrays" begin

############################################################################################

@testset "constructors" begin
    # inner constructors
    let
        p = Ptr{Int}(C_NULL)
        dp = CUDAnative.DevicePtr(p)
        CuDeviceArray{Int,1,AS.Generic}((1,), dp)
    end

    # outer constructors
    for I in [Int32,Int64]
        a = I(1)
        b = I(2)

        p = Ptr{I}(C_NULL)
        dp = CUDAnative.DevicePtr(p)

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



############################################################################################

@testset "basics" begin     # argument passing, get and setindex, length
    dims = (16, 16)
    len = prod(dims)

    @eval function array_copy(input::CuDeviceArray{Float32}, output::CuDeviceArray{Float32})
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x

        if i <= length(input)
            output[i] = Float64(input[i])   # force conversion upon setindex!
        end

        return nothing
    end

    input = round.(rand(Float32, dims) * 100)

    input_dev = CuTestArray(input)
    output_dev = CuTestArray(input)

    @cuda (1,len) array_copy(input_dev, output_dev)
    output = Array(output_dev)
    @test input ≈ output
end

@testset "iteration" begin     # argument passing, get and setindex, length
    dims = (16, 16)
    @eval function array_iteration(input::CuDeviceArray{Float32}, output::CuDeviceArray{Float32})
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        acc = 0f0
        for elem in input
            acc += elem
        end
        output[1] = acc
        return nothing
    end

    input = round.(rand(Float32, dims) * 100)

    input_dev = CuTestArray(input)
    output_dev = CuTestArray(Float32[0])

    @cuda (1, 1) array_iteration(input_dev, output_dev)
    output = Array(output_dev)
    @test sum(input) ≈ output[1]
end

############################################################################################

@testset "bounds checking" begin
    @eval function array_oob_1d(array)
        return array[1]
    end

    # NOTE: these tests verify that bounds checking is _disabled_ (see #4)

    ir = sprint(io->CUDAnative.code_llvm(io, array_oob_1d, (CuDeviceArray{Int,1,AS.Global},)))
    @test !contains(ir, "trap")

    @eval function array_oob_2d(array)
        return array[1, 1]
    end

    ir = sprint(io->CUDAnative.code_llvm(io, array_oob_2d, (CuDeviceArray{Int,2,AS.Global},)))
    @test !contains(ir, "trap")
end



############################################################################################

@testset "views" begin
    @eval function array_view(array)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x

        sub = view(array, 2:length(array)-1)
        if i <= length(sub)
            sub[i] = i
        end

        return nothing
    end

    array = zeros(Int64, 100)
    array_dev = CuTestArray(array)

    sub = view(array, 2:length(array)-1)
    for i in 1:length(sub)
        sub[i] = i
    end

    @cuda (100, 1) array_view(array_dev)
    @test array == Array(array_dev)
end

############################################################################################


@testset "bug: non-Int index to unsafe_load" begin
    @eval function array_load_index(a)
        return a[UInt64(1)]
    end

    a = [1]
    p = pointer(a)
    dp = CUDAnative.DevicePtr(p)
    da = CUDAnative.CuDeviceArray(1, dp)
    array_load_index(da)
end

end
