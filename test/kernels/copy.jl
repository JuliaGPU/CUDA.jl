@compile copy """
__global__ void copy(const float *input, float *output)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    output[i] = input[i];
}
"""

@target ptx function kernel_copy(input::CuDeviceArray{Float32}, output::CuDeviceArray{Float32})
    i = blockId_x() + (threadId_x()-1) * numBlocks_x()
    output[i] = input[i]

    return nothing
end
