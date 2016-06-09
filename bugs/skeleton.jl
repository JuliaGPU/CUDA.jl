using CUDAdrv

dev = CuDevice(0)
ctx = CuContext(dev)

@target ptx foo() = return nothing

@cuda (1,1) foo()

destroy(ctx)
