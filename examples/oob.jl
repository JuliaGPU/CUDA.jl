# Running this example under cuda-memset with DEBUG=1 properly gives line number info

# TODO: make the actual error trap at run time

using CUDAdrv, CUDAnative

dev = CuDevice(0)
ctx = CuContext(dev)

a = CuArray(Float32, 10)

function memset(a, val)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    a[i] = val
    return nothing
end

@cuda dev (1,11) memset(a, 0f0)

destroy(ctx)
