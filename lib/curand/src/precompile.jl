# handle creation
precompile(Tuple{typeof(handle_ctor), CUDACore.CuContext})

# common rand operations
precompile(Tuple{typeof(rand), Int64, Int64})
precompile(Tuple{typeof(rand), Type{Float32}, Int64, Int64})
