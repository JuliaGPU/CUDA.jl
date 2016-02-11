using CUDA

dev = CuDevice(0)
ctx = CuContext(dev)

@target ptx function kernel_crash(a::CuDeviceArray, b::CuDeviceArray)
	b[1] = length(a)
end

a = [1, 2]
b = [0]
@cuda (1, 1) kernel_crash(CuIn(a), CuOut(b))

@assert length(a) == b[1]

destroy(ctx)
