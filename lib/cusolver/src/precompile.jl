# handle creation
precompile(Tuple{typeof(dense_handle_ctor), CUDACore.CuContext})
precompile(Tuple{typeof(sparse_handle_ctor), CUDACore.CuContext})

# common factorizations
for T in (Float32, Float64)
    precompile(Tuple{typeof(getrf!), CUDACore.CuMatrix{T}})
    precompile(Tuple{typeof(geqrf!), CUDACore.CuMatrix{T}})
    precompile(Tuple{typeof(potrf!), Char, CUDACore.CuMatrix{T}})
end
