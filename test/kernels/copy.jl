@compile reference_copy """
__global__ void reference_copy(const float *input, float *output)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    output[i] = input[i];
}
"""

@target ptx function kernel_copy(input::CuDeviceArray{Float32}, output::CuDeviceArray{Float32})
    i = blockIdx_x() + (threadIdx_x()-1) * gridDim_x()
    output[i] = input[i]

    return nothing
end
