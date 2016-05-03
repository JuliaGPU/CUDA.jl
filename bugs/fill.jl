using CUDA

@target ptx function fill(start::Int, stop::Int, ca::CuDeviceArray{Int})

	start = threadIdx().x + start
	step = blockDim().x

	for i = start:step:stop
		ca[i] = i
	end

	return nothing
end

function main(args)

	n = 100
	stop = 10*n
	gpu_dst = CuArray(Int, stop)

	@cuda (1, n) fill(0, stop, gpu_dst)

	dst = to_host(gpu_dst)
	println(dst)
end

dev = CuDevice(0)
ctx = CuContext(dev)

main(ARGS)

destroy(ctx)