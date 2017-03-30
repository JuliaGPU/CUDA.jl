@testset "device arrays" begin

############################################################################################

@testset "constructors" begin
    # inner constructors
    let
        p = Ptr{Int}(C_NULL)
        @on_device CuDeviceArray{Int,1}((1,), $p)
    end

    # outer constructors
    for I in [Int32,Int64]
        a = I(1)
        b = I(2)
        p = Ptr{I}(C_NULL)

        # not parameterized
        @on_device CuDeviceArray($b, $p)
        @on_device CuDeviceArray(($b,), $p)
        @on_device CuDeviceArray(($b,$a), $p)

        # partially parameterized
        @on_device CuDeviceArray{$I}($b, $p)
        @on_device CuDeviceArray{$I}(($b,), $p)
        @on_device CuDeviceArray{$I}(($a,$b), $p)

        # fully parameterized
        @on_device CuDeviceArray{$I,1}($b, $p)
        @on_device CuDeviceArray{$I,1}(($b,), $p)
        @test_throws ErrorException @on_device CuDeviceArray{$I,1}(($a,$b), $p)
        @test_throws ErrorException @on_device CuDeviceArray{$I,2}($b, $p)
        @test_throws ErrorException @on_device CuDeviceArray{$I,2}(($b,), $p)
        @on_device CuDeviceArray{$I,2}(($a,$b), $p)
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

    input_dev = CuArray(input)
    output_dev = similar(input_dev)

    @cuda (1,len) array_copy(input_dev, output_dev)
    output = Array(output_dev)
    @test input ≈ output
end



############################################################################################

@testset "bounds checking" begin
    @eval function array_oob_1d(array)
        return array[1]
    end

    # NOTE: these tests verify that bounds checking is _disabled_ (see #4)

    ir = sprint(io->CUDAnative.code_llvm(io, array_oob_1d, (CuDeviceArray{Int,1},)))
    @test !contains(ir, "trap")

    @eval function array_oob_2d(array)
        return array[1, 1]
    end

    ir = sprint(io->CUDAnative.code_llvm(io, array_oob_2d, (CuDeviceArray{Int,2},)))
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
    array_dev = CuArray(array)

    sub = view(array, 2:length(array)-1)
    for i in 1:length(sub)
        sub[i] = i
    end

    @cuda (100, 1) array_view(array_dev)
    @test array == Array(array_dev)
end

############################################################################################

end
