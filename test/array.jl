@testset "device arrays" begin

############################################################################################

@testset "constructors" begin
    # Inner constructors
    @on_device dev CuDeviceArray{Int,1}((1,), Ptr{Int}(C_NULL))

    # Outer constructors
    @on_device dev CuDeviceArray{Int}(1, Ptr{Int}(C_NULL))
    @on_device dev CuDeviceArray{Int}((1,), Ptr{Int}(C_NULL))
    @on_device dev CuDeviceArray(1, Ptr{Int}(C_NULL))
    @on_device dev CuDeviceArray((1,), Ptr{Int}(C_NULL))
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

    @cuda dev (1,len) array_copy(input_dev, output_dev)
    output = Array(output_dev)
    @test input ≈ output

    free(input_dev)
    free(output_dev)
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

    @cuda dev (100, 1) array_view(array_dev)
    @test array == Array(array_dev)

    free(array_dev)
end

############################################################################################

end
