# CUDA Execution control

export
	launch, CuDim


get_dim_x(g::Int) = g
get_dim_x(g::(Int, Int)) = g[1]
get_dim_x(g::(Int, Int, Int)) = g[1]

get_dim_y(g::Int) = 1
get_dim_y(g::(Int, Int)) = g[2]
get_dim_y(g::(Int, Int, Int)) = g[2]

get_dim_z(g::Int) = 1
get_dim_z(g::(Int, Int)) = 1
get_dim_z(g::(Int, Int, Int)) = g[3]

typealias CuDim Union(Int, (Int, Int), (Int, Int, Int))

# Stream management

function launch(f::CuFunction, grid::CuDim, block::CuDim, args::Tuple;
	            shmem_bytes::Int=4, stream::CuStream=default_stream())
	gx = get_dim_x(grid)
	gy = get_dim_y(grid)
	gz = get_dim_z(grid)

	tx = get_dim_x(block)
	ty = get_dim_y(block)
	tz = get_dim_z(block)

	kernel_args = [ptrbox(arg) for arg in args]

	@cucall(:cuLaunchKernel, (
		Ptr{Void},  # function
		Cuint,  # grid dim x
		Cuint,  # grid dim y
		Cuint,  # grid dim z
		Cuint,  # block dim x
		Cuint,  # block dim y
		Cuint,  # block dim z
		Cuint,  # shared memory bytes, 
		Ptr{Void}, # stream 
		Ptr{Ptr{Void}}, # kernel parameters, 
		Ptr{Ptr{Void}}), # extra parameters
		f.handle, gx, gy, gz, tx, ty, tz, shmem_bytes, stream.handle, kernel_args, 0)
end

