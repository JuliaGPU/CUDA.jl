# tests related to compilation of inline CUDA C code

dev = CuDevice(0)
ctx = CuContext(dev)

architecture(dev)


## basic compilation & execution

let
    @compile dev kernel """
    __global__ void kernel()
    {
    }
    """

    @cuda (1,1) kernel()
end

@test_throws CUDAnative.CompileError let
    @compile dev kernel """
    __global__ void kernel()
    {
        invalid code
    }
    """
end

@test_throws CUDAnative.CompileError let
    @compile dev wrongname """
    __global__ void kernel()
    {
    }
    """
end


## argument passing

dims = (16, 16)
len = prod(dims)

@compile dev kernel_copy """
__global__ void kernel_copy(const float *input, float *output)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    output[i] = input[i];
}
"""

let
    input = round(rand(Float32, dims) * 100)

    input_dev = CuArray(input)
    output_dev = CuArray(Float32, dims)

    @cuda (len, 1) kernel_copy(input_dev, output_dev)
    output = Array(output_dev)
    @test input ≈ output

    free(input_dev)
    free(output_dev)
end


destroy(ctx)
