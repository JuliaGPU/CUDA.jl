extern "C" {

__device__ float add(float a, float b);

__global__ void vadd(const float *a, const float *b, float *c)
{
    int i = blockIdx.x *blockDim.x + threadIdx.x;
    c[i] = add(a[i], b[i]);
}

}
