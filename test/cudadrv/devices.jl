dev = device()

@test name(dev) isa String
@test uuid(dev) isa Base.UUID
totalmem(dev)
attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
@test warpsize(dev) == 32
capability(dev)
@grab_output show(stdout, "text/plain", dev)

@test eval(Meta.parse(repr(dev))) == dev

@test eltype(devices()) == CuDevice
@grab_output show(stdout, "text/plain", CUDA.DEVICE_CPU)
@grab_output show(stdout, "text/plain", CUDA.DEVICE_INVALID)
