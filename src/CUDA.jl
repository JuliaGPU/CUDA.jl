module CUDA

	import Base.length, Base.size
	import Base.copy!

	export

	# base
	@cucall, driver_version,

	# devices
	Device, Capability, dev_count, name, totalmem, attribute, capability,
	list_devices,

	# context
	Context, create_context, destroy, push, pop,

	# arrays
	GVector, GMatrix, free, to_host


	include("errors.jl")

	include("base.jl")
	include("devices.jl")
	include("context.jl")
	include("arrays.jl")
end