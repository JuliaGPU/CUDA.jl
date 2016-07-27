extern "C" {

__global__ void vadd(const float *a, const float *b, float *c)
{
    int i = blockIdx.x *blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

}
