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

    cudacall(kernel(), 1, 1, ())
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

@compile dev reference_copy """
__global__ void reference_copy(const float *input, float *output)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    output[i] = input[i];
}
"""

let
    input = round(rand(Float32, dims) * 100)

    input_dev = CuArray(input)
    output_dev = CuArray(Float32, dims)

    cudacall(reference_copy(), len, 1, (Ptr{Cfloat},Ptr{Cfloat}), input_dev, output_dev)
    output = Array(output_dev)
    @test_approx_eq input output

    free(input_dev)
    free(output_dev)
end


destroy(ctx)
