# handle creation
precompile(Tuple{typeof(handle_ctor), CUDACore.CuContext})

# GEMM for common types
for T in (Float32, Float64, ComplexF32, ComplexF64)
    precompile(Tuple{typeof(gemm!), Char, Char, T,
        CUDACore.CuMatrix{T}, CUDACore.CuMatrix{T},
        T, CUDACore.CuMatrix{T}})
end

# high-level matmul
for T in (Float32, Float64)
    precompile(Tuple{typeof(*),
        CUDACore.CuArray{T, 2, CUDACore.DeviceMemory},
        CUDACore.CuArray{T, 2, CUDACore.DeviceMemory}})
end
