using CUDAdrv
using Base.Test

include("library.jl")

dev = CuDevice(0)
ctx = CuContext(dev)


## basic compilation & execution

let
    @compile dev kernel """
    __global__ void kernel()
    {
    }
    """

    cudacall(kernel, 1, 1, ())
end

@test_throws CompileError let
    @compile dev kernel """
    __global__ void kernel()
    {
        invalid code
    }
    """
end

@test_throws CompileError let
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
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    output[i] = input[i];
}
"""

let
    input = round.(rand(Float32, dims) * 100)

    input_dev = CuArray(input)
    output_dev = CuArray{Float32}(dims)

    cudacall(kernel_copy, 1, len,
             Tuple{Ptr{Float32}, Ptr{Float32}},
             pointer(input_dev), pointer(output_dev))
    output = Array(output_dev)
    @test input ≈ output
end


clean_cache()   # for deterministic testing purposes
destroy(ctx)
