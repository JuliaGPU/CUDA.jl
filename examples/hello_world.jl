using CUDAdrv, CUDAnative

function hello_world()
    @cuprintf("Greetings from block %u, thread %u!\n", blockIdx().x, threadIdx().x)
    return
end

@cuda blocks=2 threads=2 hello_world()
synchronize()
