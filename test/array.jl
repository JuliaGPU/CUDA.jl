dims = (16, 16)
len = prod(dims)

@target ptx function array_copy(input, output)
    i = blockIdx().x +  (threadIdx().x-1) * gridDim().x

    output[i] = input[i]

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
