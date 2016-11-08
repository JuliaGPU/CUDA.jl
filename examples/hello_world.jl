using CUDAdrv, CUDAnative

function hello_world()
    @cuprintf("Greetings from block %d, thread %d!\n", blockIdx().x, threadIdx().x)
    return
end

dev = CuDevice(0)
ctx = CuContext(dev)

@cuda dev (2,2) hello_world()

synchronize()

destroy(ctx)
