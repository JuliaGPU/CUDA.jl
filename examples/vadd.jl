using Test

using CUDA

# generate some data on the CPU
dims = (3,4)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)
c = a .+ b

# upload to the GPU
d_a = CuArray(a)
d_b = CuArray(b)

# compute using array abstractions
d_c = d_a .+ d_b
@test c ≈ Array(d_c)

# compute using a kernel
function vadd(a, b, c)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    c[i] = a[i] + b[i]
    return
end
len = prod(dims)
@cuda threads=len vadd(d_a, d_b, d_c)
@test c ≈ Array(d_c)
