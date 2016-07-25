using CUDAdrv, CUDAnative
using Base.Test

dev = CuDevice(0)
ctx = CuContext(dev)

@compile dev kernel_vadd """
__global__ void kernel_vadd(const float *a, const float *b, float *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}
"""

dims = (3,4)
a = round(rand(Float32, dims) * 100)
b = round(rand(Float32, dims) * 100)

d_a = CuArray(a)
d_b = CuArray(b)
d_c = CuArray(Float32, dims)

len = prod(dims)
@cuda (1,len) kernel_vadd(d_a, d_b, d_c)
c = Array(d_c)
@test a+b ≈ c

destroy(ctx)
