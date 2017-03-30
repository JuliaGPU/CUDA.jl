using CUDAdrv, CUDAnative
using Base.Test

const BLOCK_SIZE = 16

function kernel_vadd(c)
    shadow = @cuStaticSharedMem(Float32, (BLOCK_SIZE,))
    tx = threadIdx().x
    c[tx] = shadow[tx]

    return nothing
end

dev = CuDevice(0)
ctx = CuContext(dev)

dims = (3,4)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)

d_a = CuArray(a)
matrix = CuArray{Float32}((BLOCK_SIZE, BLOCK_SIZE))
d_b = CuArray(b)
d_c = similar(d_a)

len = prod(dims)
CUDAnative.@code_llvm @cuda (1,len) kernel_vadd(d_c)
# c = Array(d_c)
# @test a+b ≈ c

destroy(ctx)
