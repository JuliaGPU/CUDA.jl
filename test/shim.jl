# test that the CUDA.jl shim correctly re-exports CUDACore and loads libraries

using CUDA
using cuBLAS, cuSPARSE, cuSOLVER, cuFFT, cuRAND

# CUDACore re-exports
@test CUDA.functional()
@test CuArray([1, 2, 3]) isa CuArray
@test @cuda(identity(nothing)) isa CUDACore.HostKernel

# library modules are accessible
@test CUDA.cuBLAS === cuBLAS
@test CUDA.cuSPARSE === cuSPARSE
@test CUDA.cuSOLVER === cuSOLVER
@test CUDA.cuFFT === cuFFT
@test CUDA.cuRAND === cuRAND

# CUDA.rand/randn/seed! forward to cuRAND
a = CUDA.rand(3)
@test a isa CuArray{Float32}
@test length(a) == 3

b = CUDA.randn(2, 2)
@test b isa CuArray{Float32}
@test size(b) == (2, 2)

CUDA.seed!(42)

# matmul works (cuBLAS loaded)
x = CUDA.rand(Float32, 2, 2)
y = CUDA.rand(Float32, 2, 2)
z = x * y
@test z isa CuArray{Float32}
@test size(z) == (2, 2)
