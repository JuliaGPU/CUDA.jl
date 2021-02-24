using CUDA

function kernel(a, str)
    offset = (threadIdx().x - 1) * size(a)[1]
    @cuprintf("thread %d\n", offset)
    d = CUDA.call_syscall(2, a, offset, 16)
    @cuprintf("file %d\n", d)

    for i in 1:3
        c = str[threadIdx().x, i]
        @cuprintf("String %x\n", c)
    end

    return
end

struct Foo
    x::Int
    y::Int
end

function testing3(c, d)
    println("$c - $d")
    CUDA.dump_memory(Int32, 32)
    return c-d
end

function testing2(c)
    println("TESTING2 $c")
    return
end

const cuarrays = []

function genArrays()
    a = CuArray(zeros(UInt8, 2))
    println("HERE1 $a")
    push!(cuarrays, a)

    return a
end

function kernel2()
    # @CUDA.cpu types=(Nothing, Int64,) testing2(5)
    x = @CUDA.cpu types=(Int64, Int64, Int64,) testing3(5, 45)
    ca = @CUDA.cpu types=(CuDeviceVector{UInt8, 1},) genArrays()
    ca[1] = 5
    @cuprintln("Result %d\n", x)
    return
end

function main()
    @cuda blocks=1 threads=2 kernel2()

    synchronize()
    sleep(0.1)
    flush(stdout)

    println(cuarrays)

    return
end
