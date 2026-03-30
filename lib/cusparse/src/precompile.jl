# handle creation
precompile(Tuple{typeof(handle_ctor), CUDACore.CuContext})

# sparse matrix-vector and matrix-matrix multiply
for T in (Float32, Float64)
    precompile(Tuple{typeof(mv!), Char, T,
        CuSparseMatrixCSR{T}, CUDACore.CuVector{T}, T, CUDACore.CuVector{T}, Char})
    precompile(Tuple{typeof(mm!), Char, Char, T,
        CuSparseMatrixCSR{T}, CUDACore.CuMatrix{T}, T, CUDACore.CuMatrix{T}, Char})
end
