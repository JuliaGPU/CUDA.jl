# Execution control

export
    CuDim, cudacall

immutable CuDim3
    x::Cint
    y::Cint
    z::Cint
end

# Wrapper type for conveniently specifying the dimensions
# (e.g. `(len, 2)` instead of `CuDim3(len, 2, 1)`)
typealias CuDim Union{Int, Tuple{Int, Int}, Tuple{Int, Int, Int}}
CuDim3(g::Int) = CuDim3(g, 1, 1)
CuDim3(g::Tuple{Int, Int}) = CuDim3(g[1], g[2], 1)
CuDim3(g::Tuple{Int, Int, Int}) = CuDim3(g[1], g[2], g[3])

function launch(f::CuFunction, griddim::CuDim3, blockdim::CuDim3, args::Tuple;
                shmem_bytes=0, stream::CuStream=default_stream())
    all([isbits(arg) || isa(arg, DevicePtr) for arg in args]) ||
        throw(ArgumentError("Arguments to kernel should be bitstype or device pointer"))
    args = [isa(arg, DevicePtr) ? arg.inner : arg for arg in args]  # extract DevicePtr.inner::Ptr

    # Each of kernelParams must point to a region of memory from which the actual kernel
    # parameter will be copied, hence the extra wrapping of `arg`.
    # NOTE: can't use Ref(...) here, because Ref(Ptr()) yields the inner pointer
    #       instead of the memory containing the pointer
    args = [[arg] for arg in args]

    (griddim.x>0 && griddim.y>0 && griddim.z>0)    || throw(ArgumentError("Grid dimensions should be non-null"))
    (blockdim.x>0 && blockdim.y>0 && blockdim.z>0) || throw(ArgumentError("Block dimensions should be non-null"))

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
        shmem_bytes, stream.handle, args, C_NULL)
end

# Call interface mimicking Base.ccall
function cudacall(f::CuFunction, griddim::CuDim, blockdim::CuDim, types::Tuple, values...;
                  shmem_bytes=0, stream::CuStream=default_stream())
    @assert length(types)==0 || eltype(types)==DataType # TODO: embed in type signature?

    # cconvert the values to match the kernel's signature (specified by the user)
    values = map(pair -> Base.cconvert(pair[1],pair[2]), zip(types,values))

    # TODO: tuple
    launch(f, CuDim3(griddim), CuDim3(blockdim), (values...);
           shmem_bytes=shmem_bytes, stream=stream)
end
