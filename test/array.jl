dims = (16, 16)
len = prod(dims)

## basics (argument passing, get and setindex, length)

@target ptx function array_copy(input::CuDeviceArray{Float32},
                                output::CuDeviceArray{Float32})
    i = blockIdx().x +  (threadIdx().x-1) * gridDim().x

    if i <= length(input)
        output[i] = Float64(input[i])   # force conversion upon setindex!
    end

    return nothing
end

let
    input = round(rand(Float32, dims) * 100)

    input_dev = CuArray(input)
    output_dev = CuArray(Float32, dims)

    @cuda (len, 1) array_copy(input_dev, output_dev)
    output = Array(output_dev)
    @test input ≈ output

    free(input_dev)
    free(output_dev)
end


## views

@target ptx function array_view(array)
    i = blockIdx().x +  (threadIdx().x-1) * gridDim().x

    sub = view(array, 2:length(array)-1)
    if i <= length(sub)
        sub[i] = i
    end

    return nothing
end

let
    array = zeros(Int64, 100)
    array_dev = CuArray(array)

    sub = view(array, 2:length(array)-1)
    for i in 1:length(sub)
        sub[i] = i
    end

    @cuda (100, 1) array_view(array_dev)
    @test array == Array(array_dev)

    free(array_dev)
end

