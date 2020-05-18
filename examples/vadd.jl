using Test

using CUDA

function vadd(a, b, c)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    c[i] = a[i] + b[i]
    return
end

dims = (3,4)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)
c = similar(a)

d_a = CuArray(a)
d_b = CuArray(b)
d_c = CuArray(c)

len = prod(dims)
@cuda threads=len vadd(d_a, d_b, d_c)
c = Array(d_c)
@test a+b ≈ c
