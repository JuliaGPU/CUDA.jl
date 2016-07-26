using CUDAdrv, CUDAnative
using Base.Test

@target ptx function kernel_vadd(a, b, c)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    c[i] = a[i] + b[i]

    return nothing
end

dev = CuDevice(0)
ctx = CuContext(dev)

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
