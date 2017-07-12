using CUDAdrv, CUDAnative

function hello_world()
    @cuprintf("Greetings from block %u, thread %u!\n", blockIdx().x, threadIdx().x)
    return
end

@cuda (2,2) hello_world()
synchronize()
