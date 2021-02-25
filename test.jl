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


## Linked List trial

abstract type List{T}
end

mutable struct Nil{T} <: List{T}
end

mutable struct Cons{T} <: List{T}
    value::T
    next::List{T} # problem
end

struct Nil2{T}
end

struct Cons2{T, TT}
    value::T
    next::TT
end


Cons{T}(value::T) where T = Cons{T}(value, Nil{T}())

function List{T}(pointer, count::Integer) where T
    result ::Union{Cons{Int64}, Nil{Int64}} = Nil{T}()
    while count > 1
        result = Cons{T}(pointer[count], result)
        pointer[count] = 0
        count -= 1
    end
    # for i in count:-1:1
    #     # CUDA.@cpu types=(Nothing, Int) printIt(i)
    #     result = Cons{T}(pointer[i], result)
    # end
    result
end

function listR(pointer, count::Integer)
    result ::Union{Cons{Int64}, Nil{Int64}} = Nil{Int64}()
    while count > 1
        result = Cons{Int64}(@inbounds pointer[count], result)
        @inbounds pointer[count] = 0
        count -= 1
    end
    # for i in count:-1:1
    #     # CUDA.@cpu types=(Nothing, Int) printIt(i)
    #     result = Cons{T}(pointer[i], result)
    # end
    return
end


function kernels(a)
    b = Nil2{Int64}()
    c = Cons2{Int64, Nil2{Int64}}(@inbounds a[1], b)
    d = Cons2{Int64, Cons2{Int64, Nil2{Int64}}}(@inbounds a[2], c)

    a[3] = d.next.value

    return
end

function main()
    b = Array(1:21)
    a = CuArray(b)

    @CUDA.sync blocking=false @CUDA.cuda kernels(a)

    println(a)

    return
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

# cu_area = Val("cpucall_area")
getFoobar20(x) = FooBar(x, 42)
function test_test(x)

    function testFooBars()
        # @cuprintln("%d", CUDA.hostcall_area())
        # b = CUDA.hostcall_area()
        size = warpsize()
        id = threadIdx().x - 1
        block = blockIdx().x - 1
        warp = get_warp_id()
        # @cuprintln("id $id block $block size $size warp $warp")

        foobar = @CUDA.cpu types=(CUDA.HostRef, Int64,) getFoobar20(id)
        @CUDA.cpu types=(Nothing, CUDA.HostRef,) printIt(foobar)
        # if id % size == 0
            @CUDA.cpu types=(Nothing, Int128, Int64) printIt(id, block)
        # end

        return
    end

    # CUDA.@sync
    @cuda threads=x testFooBars()
    return
end


function kernel3(a)
    a[1] = 42
    return
end

function test_view()
    zs = zeros(20)
    cu_zs = CuArray(zs)

    CUDA.@sync @cuda kernel3(view(cu_zs, 5))

    println(cu_zs)
end


## Benchmark cuda on GPU/CPU

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
    # for n in 1:10:201
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
    # println(result)
end
