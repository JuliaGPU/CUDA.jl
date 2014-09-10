module CUDA

	import Base.length, Base.size, Base.eltype, Base.ndims
	import Base.copy!

	export

	# errors
	CuDriverError, description,

	# base
	@cucall, driver_version,

	# devices
	CuDevice, CuCapability, dev_count, name, totalmem, attribute, capability,
	list_devices,

	# context
	CuContext, destroy, push, pop,

	# module
	CuModule, CuFunction, unload,
	CuGlobal, get, set,

	# stream
	CuStream, synchronize,

	# execution
	launch, CuDim,

	# arrays
	CuPtr, CuArray, free, to_host, ndims,

	# native
	threadId_x, threadId_y, threadId_z,
	numThreads_x, numThreads_y, numThreads_z,
	blockId_x, blockId_y, blockId_z,
	numBlocks_x, numBlocks_y, numBlocks_z,
	warpsize, sync_threads,
	CuIn, CuOut, CuInOut,
	cuSharedMem, setCuSharedMem, getCuSharedMem,
	@cuda,

	# math
	sin, cos, floor


	include("errors.jl")

	include("base.jl")
	include("devices.jl")
	include("context.jl")
	include("module.jl")
	include("stream.jl")
	include("execution.jl")

	include("arrays.jl")

	include("native.jl")
	include("math.jl")
	include("profile.jl")
end