# Execution control

export
    launch, CuDim


typealias CuDim Union(Int, (Int, Int), (Int, Int, Int))
dim3(g::Int) = (g, 1, 1)
dim3(g::(Int, Int)) = (g[1], g[2], 1)
dim3(g::(Int, Int, Int)) = g

function launch(f::CuFunction, grid::CuDim, block::CuDim, args::Tuple;
                shmem_bytes::Int=4, stream::CuStream=default_stream())
	griddim = dim3(grid)
	blockdim = dim3(block)

    @assert all([isbits(arg) || isa(arg, DevicePtr) for arg in args])
    kernel_args = [[arg] for arg in args]

    @cucall(:cuLaunchKernel, (
        Ptr{Void},  			# function
        Cuint, Cuint, Cuint,  	# grid dimensions (x, y, z)
        Cuint, Cuint, Cuint,  	# block dimensions (x, y, z)
        Cuint,  				# shared memory bytes, 
        Ptr{Void}, 				# stream 
        Ptr{Ptr{Void}}, 		# kernel parameters
        Ptr{Ptr{Void}}), 		# extra parameters
        f.handle, griddim[1], griddim[2], griddim[3], blockdim[1], blockdim[2],
        blockdim[3], shmem_bytes, stream.handle, kernel_args, C_NULL)
end
