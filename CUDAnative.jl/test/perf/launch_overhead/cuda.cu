extern "C" __global__ void kernel_dummy(float *ptr)
{
    ptr[blockIdx.x] = 0;
}
