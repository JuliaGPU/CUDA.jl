using CUDA
using BenchmarkTools
using GPUCompiler
using LLVM
using LLVM.Interop
using Core: LLVMPtr


## UTIL

function printIt(a...)
    println("it $a")
    return
end


## Linked List

mutable struct Nil end
mutable struct LinkedList{T}
    value::T
    next::Union{Nil, LinkedList{T}}
end

function linked_list_from_list(l, count, ::Type{T})::Union{Nil, LinkedList{Int}} where {T}
    if count == 0
        return Nil
    end
    tail :: LinkedList{T} = LinkedList(l[1], Nil())
    for i in 2:count
        t = l[i]
        tail = LinkedList(t, tail)
    end
    return tail
end

function count_list(list::Union{Nil, LinkedList{Int}})
    sum = 0
    while isa(list, LinkedList{Int})
        sum += list.value
        list = list.next
    end
    return sum
end

function test()
    function ll_kernel(a, c)
        l = linked_list_from_list(a, c, Int)
        d = count_list(l)
        @cuprintln("It %d", d)
        # @CUDA.cpu types=(Nothing, Int) printIt()
        return
    end
    a = [1,2,3,4,5]
    c = CuArray(a)
    @cuda ll_kernel(c, 5)
end


## Features trial (view and HostRef)
mutable struct FooBar
    x::Int
    y::Int
end

@inline function get_thread_id()
    return (blockIdx().x - 1) * blockDim().x + threadIdx().x
end

@inline function get_warp_id()
    return div(get_thread_id() - 1, warpsize()) + 1
end

getFoobar20(x) = FooBar(x, 42)
function test_test(x)

    function testFooBars()
        size = warpsize()
        id = threadIdx().x - 1
        block = blockIdx().x - 1
        warp = get_warp_id()

        foobar = @CUDA.cpu types=(CUDA.HostRef, Int64,) getFoobar20(id) # Test HostRef
        @CUDA.cpu types=(Nothing, CUDA.HostRef,) printIt(foobar)
        @CUDA.cpu types=(Nothing, Int128, Int64) printIt(id, block)

        return
    end

    CUDA.@sync @cuda threads=x testFooBars()
    return
end


function kernel3(a)
    nanosleep(UInt32(18))
    a[1] = 42
    return
end

function test_view()
    zs = zeros(20)
    cu_zs = CuArray(zs)

    CUDA.@sync @cuda kernel3(view(cu_zs, 5))

    println(cu_zs)
end


## Benchmark bump allocator on GPU/CPU

mutable struct Bump
    ptr::Ptr{UInt8}
    at::Int64
    size::Int64
end

cpu_space = Vector{UInt8}(undef, 1024)
gpu_space = CUDA.Mem.alloc(Mem.Device, 1024)
gpu_bumper = Bump(reinterpret(Ptr{UInt8}, gpu_space.ptr), 0, 1024)
cpu_bumper = Bump(pointer(cpu_space), 0, 1024)

function reset_bumper!(bump::Bump)
    println("resetting bumper")
    bump.at = 0

    return
end

function bump_malloc(sizet::Int64, bump::Bump)::Ptr{UInt8}
    if (bump.at + sizet) >= bump.size
        bump.at = 0
    end

    ptr = bump.ptr + bump.at
    bump.at += sizet

    ptr
end

gpu_bump_malloc(sizet) = bump_malloc(convert(Int64, sizet), gpu_bumper)
cpu_bump_malloc(sizet) = bump_malloc(convert(Int64, sizet), cpu_bumper)


macro async_benchmarkable(ex...)
    quote
        # use non-blocking sync to reduce overhead
        @benchmarkable (CUDA.@sync blocking=false $(ex...))
    end
end


function cuda_malloc()
    a = reinterpret(Ptr{UInt8}, malloc(convert(UInt64, 1)))
    unsafe_store!(a, 42)
    return
end


function gpu_malloc()
    a = @CUDA.cpu types=(Ptr{UInt8}, Int64,) gpu_bump_malloc(convert(Int64, 1))
    unsafe_store!(a, 42)
    return
end

function cpu_malloc(n)
    for x in 1:n
        a = cpu_bump_malloc(1)
        unsafe_store!(a, 42)
    end
end


function bench2()
    BenchmarkTools.DEFAULT_PARAMETERS.samples = 50
    SUITE = BenchmarkGroup()

    for n in 1:10:200
        ns = string(n)
        SUITE[ns] = BenchmarkGroup()

        # eval(quote $SUITE[$ns]["cuda"] = @async_benchmarkable @cuda threads=$n cuda_malloc() end)
        eval(quote $SUITE[$ns]["gpu"] = @benchmarkable (CUDA.@sync blocking=false @cuda threads=$n gpu_malloc()) setup=reset_bumper!(gpu_bumper) end)
        eval(quote $SUITE[$ns]["cpu"] = @benchmarkable cpu_malloc($n) setup=reset_bumper!(cpu_bumper) end)
    end

    warmup(SUITE; verbose=false)
    result = median(run(SUITE, verbose=false))

    println("threads,gpu,cpu,cuda")
    # for n in 1:10:201
    for n in 1:10:200
        xs = result[string(n)]
        # cuda = xs["cuda"]
        gpu = xs["gpu"]
        cpu = xs["cpu"]

        # print(cuda)
        println("$n,$(time(gpu)/1000000),$(time(cpu)/1000000),$(0/1000000)")
    end
end


function small_bench()
    result = nothing
    timer = CUDA.time_it() do timer
        result = @benchmark test_test(100) setup=(CUDA.start_sample!($timer))
    end

    m = time(mean(result)) / 1000000000

    println("mean $m secs timer $timer")
end

function bench()
    BenchmarkTools.DEFAULT_PARAMETERS.samples = 50
    SUITE = BenchmarkGroup()

    # for n in 1:10:
    for n in 1:30
        ns = string(n)
        SUITE[ns] = BenchmarkGroup()

        if n < 101
            eval(quote $SUITE[$ns]["cuda"] = @async_benchmarkable @cuda threads=$n cuda_malloc() end)
        end

        eval(quote $SUITE[$ns]["cpu"] = @async_benchmarkable @cuda threads=$n cpu_malloc() end)
        eval(quote $SUITE[$ns]["nothing"] = @async_benchmarkable @cuda threads=$n nothing_kernel() end)
    end

    warmup(SUITE; verbose=false)
    result = median(run(SUITE, verbose=false))

    println("threads,cuda,cpu,nothing")
    for n in 1:30
        xs = result[string(n)]
        cpu = xs["cpu"]
        empty = xs["nothing"]

        # print(cuda)
        if n < 101
            cuda = xs["cuda"]
            println("$n,$(time(cuda)/1000000),$(time(cpu)/1000000),$(time(empty)/1000000)")
        else
            println("$n,,$(time(cpu)/1000000),$(time(empty)/1000000)")
        end
    end
end


# BenchmarkTools.Trial:
#   memory estimate:  461.59 KiB
#   allocs estimate:  23388
#   --------------
#   minimum time:     40.987 ms (0.00% GC)
#   median time:      44.807 ms (0.00% GC)
#   mean time:        45.837 ms (0.30% GC)
#   maximum time:     60.897 ms (24.97% GC)
#   --------------
#   samples:          110
#   evals/sample:     1

# BenchmarkTools.Trial:
#   memory estimate:  238.42 KiB
#   allocs estimate:  7586
#   --------------
#   minimum time:     646.981 ms (0.00% GC)
#   median time:      651.066 ms (0.00% GC)
#   mean time:        650.383 ms (0.00% GC)
#   maximum time:     653.093 ms (0.00% GC)
#   --------------
#   samples:          8
#   evals/sample:     1
