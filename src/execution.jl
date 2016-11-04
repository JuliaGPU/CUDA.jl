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

"""
Launch a CUDA function on a GPU.

This is a low-level call, prefer to use `cudacall` instead.
"""
function launch(f::CuFunction, griddim::CuDim3, blockdim::CuDim3,
                shmem::Int, stream::CuStream, args)
    all([isbits(arg) || isa(arg, DevicePtr) for arg in args]) ||
        throw(ArgumentError("Arguments to kernel should be bitstype or device pointer"))
    args = [isa(arg, DevicePtr) ? arg.ptr : arg for arg in args]  # extract inner Ptr

    # If f has N parameters, then kernelParams needs to be an array of N pointers.
    # Each of kernelParams[0] through kernelParams[N-1] must point to a region of memory
    # from which the actual kernel parameter will be copied.
    argptrs = Ptr{Void}[Base.pointer_from_objref(arg) for arg in args]

    (griddim.x>0 && griddim.y>0 && griddim.z>0)    || throw(ArgumentError("Grid dimensions should be non-null"))
    (blockdim.x>0 && blockdim.y>0 && blockdim.z>0) || throw(ArgumentError("Block dimensions should be non-null"))

    @apicall(:cuLaunchKernel, (
        CuFunction_t,  			# function
        Cuint, Cuint, Cuint,  	# grid dimensions (x, y, z)
        Cuint, Cuint, Cuint,  	# block dimensions (x, y, z)
        Cuint,  				# shared memory bytes,
        CuStream_t,				# stream
        Ptr{Ptr{Void}}, 		# kernel parameters
        Ptr{Ptr{Void}}), 		# extra parameters
        f,
        griddim.x, griddim.y, griddim.z,
        blockdim.x, blockdim.y, blockdim.z,
        shmem, stream, argptrs, C_NULL)
end

"""
ccall-like interface to launching a CUDA function on a GPU
"""
function cudacall(f::CuFunction, griddim::CuDim, blockdim::CuDim, types, values...;
                  shmem=0, stream::CuStream=CuDefaultStream())
    tt = Base.to_tuple_type(types)

    # convert the values to match the kernel's signature (specified by the user)
    values = map(pair -> Base.unsafe_convert(pair[1], Base.cconvert(pair[1],pair[2])),
                 zip(tt.parameters,values))

    launch(f, CuDim3(griddim), CuDim3(blockdim), shmem, stream, values)
end
