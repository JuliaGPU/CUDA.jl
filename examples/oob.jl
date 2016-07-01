# Running this example under cuda-memset with DEBUG=1 properly gives line number info

# TODO: make the actual error trap at run time

using CUDAdrv, CUDAnative

dev = CuDevice(0)
ctx = CuContext(dev)

a = CuArray(Float32, 10)

@target ptx function memset(a, val)
    i = blockIdx().x +  (threadIdx().x-1) * gridDim().x
    a[i] = 0f0
    return nothing
end

@cuda (11,1) memset(a, 0f0)

destroy(ctx)
