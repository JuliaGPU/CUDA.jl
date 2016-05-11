# Execution control

export
    launch, CuDim

immutable dim3
  x::Int
  y::Int
  z::Int
end

# Wrapper type for conveniently specifying the dimensions
# (e.g. `(len, 2)` instead of `dim3(len, 2, 1)`)
typealias CuDim Union{Int, Tuple{Int, Int}, Tuple{Int, Int, Int}}
dim3(g::Int) = dim3(g, 1, 1)
dim3(g::Tuple{Int, Int}) = dim3(g[1], g[2], 1)
dim3(g::Tuple{Int, Int, Int}) = dim3(g[1], g[2], g[3])

function launch(f::CuFunction, grid::CuDim, block::CuDim, args::Tuple;
                shmem_bytes::Int=0, stream::CuStream=default_stream())
	griddim = dim3(grid)
	blockdim = dim3(block)

    all(dim->(dim > 0), grid)  || throw(ArgumentError("Grid dimensions should be non-null"))
    all(dim->(dim > 0), block) || throw(ArgumentError("Block dimensions should be non-null"))

    all([isbits(arg) || isa(arg, DevicePtr) for arg in args]) ||
        throw(ArgumentError("Arguments to kernel should be bitstype or device pointer"))
    kernel_args = Any[[arg] for arg in args]

    @cucall(:cuLaunchKernel, (
        Ptr{Void},  			# function
        Cuint, Cuint, Cuint,  	# grid dimensions (x, y, z)
        Cuint, Cuint, Cuint,  	# block dimensions (x, y, z)
        Cuint,  				# shared memory bytes,
        Ptr{Void}, 				# stream
        Ptr{Ptr{Void}}, 		# kernel parameters
        Ptr{Ptr{Void}}), 		# extra parameters
        f.handle,
        griddim.x, griddim.y, griddim.z,
        blockdim.x, blockdim.y, blockdim.z,
        shmem_bytes, stream.handle, kernel_args, C_NULL)
end
