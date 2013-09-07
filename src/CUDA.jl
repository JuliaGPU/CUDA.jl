module CUDA

	export

	# base
	@cucall, driver_version,

	# devices
	Device, Capability, dev_count, name, totalmem, attribute, capability,
	list_devices,

	# context
	Context, create_context, destroy, push, pop


	include("errors.jl")

	include("base.jl")
	include("devices.jl")
	include("context.jl")
end