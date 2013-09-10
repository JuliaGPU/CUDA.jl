# A simple example to demonstrate the use of CUDA.jl

using CUDA

try

	println("Devices")
	list_devices()
	println()

	dev = Device(0)
	println("create context")
	ctx = create_context(dev)

	println("load module from vadd.ptx")
	md = CuModule("vadd.ptx")

	println("get function vadd")
	f = CuFunction(md, "vadd")

	# a = rand(12)
	# g = GVector(a)
	# a2 = to_host(g)

	# println("a = $a")
	# println("a2 = $a2")
	# println("a == a2 ? $(a == a2)")

	# free(g)

	println("unload module")
	unload(md)

	println("destroy context")
	destroy(ctx)

catch err

	if isa(err, CuDriverError)
		println("$err: $(description(err))")
	else
		throw(err)
	end
end

