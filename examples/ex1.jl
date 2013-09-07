# A simple example to demonstrate the use of CUDA.jl

using CUDA


println("Devices")
list_devices()
println()

dev = Device(0)
println("create context")
ctx = create_context(dev)


println("destroy context")
destroy(ctx)
