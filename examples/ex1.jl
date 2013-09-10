# A simple example to demonstrate the use of CUDA.jl

using CUDA


println("Devices")
list_devices()
println()

dev = Device(0)
println("create context")
ctx = create_context(dev)

a = rand(12)
g = GVector(a)
a2 = to_host(g)

println("a = $a")
println("a2 = $a2")
println("a == a2 ? $(a == a2)")

free(g)

println("destroy context")
destroy(ctx)
