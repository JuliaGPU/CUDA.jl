# A simple example to demonstrate the use of CUDA.jl

using CUDA

try

	println("Devices")
	list_devices()
	println()

	dev = CuDevice(0)
	println("create context")
	ctx = create_context(dev)

	println("load module from vadd.ptx")
	md = CuModule("vadd.ptx")

	println("get function vadd")
	f = CuFunction(md, "vadd")

	siz = (3, 4)
	len = prod(siz)

	println("load array a to GPU")
	a = round(rand(Float32, siz) * 100)
	ga = CuArray(a)

	println("load array b to GPU")
	b = round(rand(Float32, siz) * 100)
	gb = CuArray(b)

	println("create array c on GPU")
	gc = CuArray(Float32, siz)

	println("launch kernel")
	launch(f, len, 1, (ga, gb, gc))

	println("fetch results from GPU")
	c = to_host(gc)

	println("free GPU memory")
	free(ga)
	free(gb)
	free(gc)

	println("Results:")
	println("a = \n$a")
	println("b = \n$b")
	println("c = \n$c")

	println("c == (a + b) ? $(c == (a + b))")

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

