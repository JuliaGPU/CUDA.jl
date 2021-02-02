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

function testing(c, d)
    println("TESTING $c $d")
    return c-d
end

function testing2(c)
    println("TESTING2 $c")
end

function kernel2()
    x = @CUDA.cpu types=(Int64, Int64,Int64,) testing(5, 45)
    @cuprintln("HEre %d\n", x)
    return
end

function main()
    width = 16
    files = ["artifacts.toml", "ARTIFACTS.toml"]
    d = zeros(UInt8, width, size(files)[1])

    for (i, file) in enumerate(files)
        offset = ((i-1)*width) + 1
        copyto!(d, offset, Vector{UInt8}(file))
    end

    # d = zeros(Int, 32)
    cu_d = CuArray(d)

    @cuda blocks=1 threads=1 kernel2()

    sleep(0.4)
    return
end
