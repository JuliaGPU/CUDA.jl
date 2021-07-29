
function maybePrint(a)
    # println(a)
    write(stdout, string(a))
end

function printIt(a...)
    maybePrint(a)
    return
end


function print_it_not(a...)
    write(devnull, string(a))
    return
end



function gpu_println_blocking()
    for i in 1:100
        id = threadIdx().x - 1
        block = blockIdx().x

        @CUDA.cpu blocking=true types=(Nothing, Int64, Int64) printIt(id, block)

        a = 6.0
        for i in 0:100
            a = cos(a)
        end

        @CUDA.cpu blocking=true types=(Nothing, Int64, Int64) printIt(id, 4)
    end

    return
end

function gpu_println_blocking_not()
    for i in 1:100
        id = threadIdx().x - 1
        block = blockIdx().x

        @CUDA.cpu blocking=true types=(Nothing, Int64, Int64) print_it_not(id, block)

        a = 6.0
        for i in 0:100
            a = cos(a)
        end

        @CUDA.cpu blocking=true types=(Nothing, Int64, Int64) print_it_not(id, 4)
    end

    return
end

function gpu_println_non_blocking()
    for i in 1:100
        id = threadIdx().x - 1
        block = blockIdx().x

        @CUDA.cpu blocking=false types=(Nothing, Int64, Int64) printIt(id, block)

        a = 6.0
        for i in 0:100
            a = cos(a)
        end

        @CUDA.cpu blocking=false types=(Nothing, Int64, Int64) printIt(id, 4)
    end

    return
end


function gpu_println_non_blocking_not()
    for i in 1:100
        id = threadIdx().x - 1
        block = blockIdx().x

        @CUDA.cpu blocking=false types=(Nothing, Int64, Int64) print_it_not(id, block)

        a = 6.0
        for i in 0:100
            a = cos(a)
        end

        @CUDA.cpu blocking=false types=(Nothing, Int64, Int64) print_it_not(id, 4)
    end

    return
end

function cuda_println()
    for i in 1:100
        id = threadIdx().x - 1
        block = blockIdx().x

        @cuprintln("$id, $block")

        a = 6.0
        for i in 0:100
            a = cos(a)
        end

        @cuprintln("$id, 3")
    end

    return
end


function print_bench(file=nothing)
    if file === nothing
        file = "tmp/printbench/test1.csv"
    end

    open(file, "w") do io
        perform_benchmarks([64, 128, 256], [1, 8], [
            SimpleRunner("cuda", (ns, bs) -> (() -> @sync @cuda threads=ns blocks=bs policy_type=CUDA.NoPolicy cuda_println())),
            SimpleRunner("gpu_simple_blocking", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager gpu_println_blocking())),
                quote (ns, bs) -> (CUDA.SimpleAreaManager(256, 100),) end),
            SimpleRunner("gpu_simple_non_blocking", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager gpu_println_non_blocking())),
                quote (ns, bs) -> (CUDA.SimpleAreaManager(256, 100),) end),
            SimpleRunner("gpu_warp_blocking", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager gpu_println_blocking())),
                quote (ns, bs) -> (CUDA.WarpAreaManager(8, 100),) end),
            SimpleRunner("gpu_warp_non_blocking", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager gpu_println_non_blocking())),
                quote (ns, bs) -> (CUDA.WarpAreaManager(8, 100),) end),

            SimpleRunner("gpu_simple_blocking_2", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager poller_count=2 gpu_println_blocking())),
                quote (ns, bs) -> (CUDA.SimpleAreaManager(256, 100),) end),
            SimpleRunner("gpu_simple_non_blocking_2", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager poller_count=2 gpu_println_non_blocking())),
                quote (ns, bs) -> (CUDA.SimpleAreaManager(256, 100),) end),
            SimpleRunner("gpu_warp_blocking_2", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager poller_count=2 gpu_println_blocking())),
                quote (ns, bs) -> (CUDA.WarpAreaManager(8, 100),) end),
            SimpleRunner("gpu_warp_non_blocking_2", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager poller_count=2 gpu_println_non_blocking())),
                quote (ns, bs) -> (CUDA.WarpAreaManager(8, 100),) end),

            SimpleRunner("gpu_simple_blocking_4", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager poller_count=4 gpu_println_blocking())),
                quote (ns, bs) -> (CUDA.SimpleAreaManager(256, 100),) end),
            SimpleRunner("gpu_simple_non_blocking_4", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager poller_count=4 gpu_println_non_blocking())),
                quote (ns, bs) -> (CUDA.SimpleAreaManager(256, 100),) end),
            SimpleRunner("gpu_warp_blocking_4", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager poller_count=4 gpu_println_blocking())),
                quote (ns, bs) -> (CUDA.WarpAreaManager(8, 100),) end),
            SimpleRunner("gpu_warp_non_blocking_4", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager poller_count=4 gpu_println_non_blocking())),
                quote (ns, bs) -> (CUDA.WarpAreaManager(8, 100),) end),

        ]; io = io, samples=3, evals=1, measures=[:median, :std])
    end
end


# Same as print_bench but without actually printing
function print_bench_2(file=nothing)
    if file === nothing
        file = "tmp/printbench/test_2_1.csv"
    end

    open(file, "w") do io
        perform_benchmarks([64, 128, 256], [1, 8], [
            SimpleRunner("cuda", (ns, bs) -> (() -> @sync @cuda threads=ns blocks=bs policy_type=CUDA.NoPolicy cuda_println())),
            SimpleRunner("gpu_simple_blocking", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager gpu_println_blocking_not())),
                quote (ns, bs) -> (CUDA.SimpleAreaManager(256, 100),) end),
            SimpleRunner("gpu_simple_non_blocking", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager gpu_println_non_blocking_not())),
                quote (ns, bs) -> (CUDA.SimpleAreaManager(256, 100),) end),
            SimpleRunner("gpu_warp_blocking", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager gpu_println_blocking_not())),
                quote (ns, bs) -> (CUDA.WarpAreaManager(8, 100),) end),
            SimpleRunner("gpu_warp_non_blocking", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager gpu_println_non_blocking_not())),
                quote (ns, bs) -> (CUDA.WarpAreaManager(8, 100),) end),

            SimpleRunner("gpu_simple_blocking_2", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager poller_count=2 gpu_println_blocking_not())),
                quote (ns, bs) -> (CUDA.SimpleAreaManager(256, 100),) end),
            SimpleRunner("gpu_simple_non_blocking_2", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager poller_count=2 gpu_println_non_blocking_not())),
                quote (ns, bs) -> (CUDA.SimpleAreaManager(256, 100),) end),
            SimpleRunner("gpu_warp_blocking_2", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager poller_count=2 gpu_println_blocking_not())),
                quote (ns, bs) -> (CUDA.WarpAreaManager(8, 100),) end),
            SimpleRunner("gpu_warp_non_blocking_2", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager poller_count=2 gpu_println_non_blocking_not())),
                quote (ns, bs) -> (CUDA.WarpAreaManager(8, 100),) end),

            SimpleRunner("gpu_simple_blocking_4", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager poller_count=4 gpu_println_blocking_not())),
                quote (ns, bs) -> (CUDA.SimpleAreaManager(256, 100),) end),
            SimpleRunner("gpu_simple_non_blocking_4", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager poller_count=4 gpu_println_non_blocking_not())),
                quote (ns, bs) -> (CUDA.SimpleAreaManager(256, 100),) end),
            SimpleRunner("gpu_warp_blocking_4", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager poller_count=4 gpu_println_blocking_not())),
                quote (ns, bs) -> (CUDA.WarpAreaManager(8, 100),) end),
            SimpleRunner("gpu_warp_non_blocking_4", (ns, bs) -> ((manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager poller_count=4 gpu_println_non_blocking_not())),
                quote (ns, bs) -> (CUDA.WarpAreaManager(8, 100),) end),

        ]; io = io, samples=3, evals=1, measures=[:median, :std])
    end
end
