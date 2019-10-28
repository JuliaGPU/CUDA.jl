using CUDA

function hello_world()
    @cuprintln("Greetings from block $(blockIdx().x), thread $(threadIdx().x)!")
    return
end

@cuda blocks=2 threads=2 hello_world()
synchronize()
