using CUDAdrv

dev = CuDevice(0)
ctx = CuContext(dev)

@target ptx function foo(a::CuDeviceArray)
    b = "test"
    a[1] = length(b)
    return nothing
end

x = [0]
@cuda (1,1) foo(CuOut(x))

destroy(ctx)
