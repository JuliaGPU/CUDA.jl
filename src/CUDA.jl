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
	CuContext, create_context, destroy, push, pop,

	# module
	CuModule, CuFunction, unload,

	# stream
	CuStream, synchronize,

	# execution
	launch, CuDim,

	# arrays
	CuPtr, CuArray, free, to_host, ndims,

	# native
	threadId_x, threadId_y, threadId_z,
	CuIn, CuOut, CuInOut,
	@cuda


	include("errors.jl")

	include("base.jl")
	include("devices.jl")
	include("context.jl")
	include("module.jl")
	include("stream.jl")
	include("execution.jl")

	include("arrays.jl")

	include("native.jl")
end