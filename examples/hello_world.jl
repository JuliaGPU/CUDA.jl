using CUDAdrv, CUDAnative

hello_world() =
    @cuprintf("Greetings from block %u, thread %u!\n", blockIdx().x, threadIdx().x)

@cuda blocks=2 threads=2 hello_world()
synchronize()
