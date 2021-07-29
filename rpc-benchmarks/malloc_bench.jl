if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg

    if isfile("Project.toml")
        Pkg.activate("./")
    end
end

using CUDA
using BenchmarkTools
using LLVM
using LLVM.Interop
using Core: LLVMPtr


## Benchmark bump allocator on GPU/CPU

mutable struct Bump
    ptr::Ptr{UInt8}
    at::Int64
    size::Int64
end

cpu_space = Vector{UInt8}(undef, 1024)
gpu_space = nothing
gpu_bumper = nothing
cpu_bumper = Bump(pointer(cpu_space), 0, 1024)

function realloc_gpu_bumper(size)
    global gpu_space
    global gpu_bumper

    gpu_space = CUDA.Mem.alloc(Mem.Device, size)
    print(gpu_space)
    gpu_bumper = Bump(reinterpret(Ptr{UInt8}, gpu_space.ptr), 0, size)
end

function reset_bumper!(bump::Bump)
    bump.at = 0
    return
end

function bump_malloc(sizet::Int64, bump::Bump)::Ptr{UInt8}
    # println("mallocing $sizet")
    if (bump.at + sizet) >= bump.size
        bump.at = 0
    end

    ptr = bump.ptr + bump.at
    bump.at += sizet

    ptr
end

gpu_bump_malloc(sizet, bumper=gpu_bumper) = bump_malloc(convert(Int64, sizet), bumper)
cpu_bump_malloc(sizet) = bump_malloc(convert(Int64, sizet), cpu_bumper)


success_mallocs = 0
function in_malloc(x)
    global success_mallocs
    success_mallocs += x
    return
end

# This takes a second
function print_maximum_cuda_mallocs(chunks)
    global success_mallocs
    success_mallocs = 0

    function kernel()
        while true
            a = malloc(UInt64(chunks))
            a == null_ptr(Cvoid) && break
            @CUDA.cpu types=(Nothing, Int64) in_malloc(chunks)
        end
        return
    end

    @sync @cuda kernel()

    println("Total alloced size $success_mallocs in $(success_mallocs / chunks) mallocs")
    println("$success_mallocs total_size")
end


null_ptr(::Type{T}) where {T} = reinterpret(Ptr{T}, C_NULL)
function cuda_malloc()
    for _ in 1:50
        a = reinterpret(Ptr{UInt8}, malloc(UInt64(32)))
        if a != null_ptr(UInt8)
            unsafe_store!(a, 44)
        else
            @cuprint("No $(blockIdx().x) $(threadIdx().x)")
            break
        end
    end
    return
end


function single_cuda_malloc(threads, blocks, manager, policy)
    @sync @cuda threads=1 blocks=1 cuda_malloc()
    @time @sync @cuda threads=threads blocks=blocks cuda_malloc()
end


function gpu_malloc()
    for _ in 1:50
        a = @CUDA.cpu types=(Ptr{UInt8}, Int64,) gpu_bump_malloc(32)
        unsafe_store!(a, 44)
    end
    return
end



# T -> T -> (S, T)
add_with_state2(v1, v2) = (v1[1], (v1[1]+v2[1],))
# S -> R -> (R, R)
get_with_state2(state, t) = (t, t+state)

function gpu_malloc_gather()
    for _ in 1:50
        a = @CUDA.cpu types=(Ptr{UInt8}, Int64,) gran_gather=add_with_state2 gran_scatter=get_with_state2 gpu_bump_malloc(convert(Int64, 32))
        unsafe_store!(a, 44)
    end
    return
end


function cpu_malloc(n)
    for x in 1:n
        a = cpu_bump_malloc(1)
        unsafe_store!(a, 42)
    end
end


function malloc_bench(file=nothing)
    if file === nothing
        file = "tmp/test3.csv"
    end

    runners = [
        SimpleRunner("warp_gpu", (ns, bs) -> (
            (manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager gpu_malloc())),
            quote (ns, bs) -> (CUDA.WarpAreaManager(8, 100), ) end
        ),
        SimpleRunner("warp_gpu_gather", (ns, bs) -> (
            (manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager shmem=2*ns*sizeof(Int64) gpu_malloc_gather())),
            quote (ns, bs) -> (CUDA.WarpAreaManager(8, 100), ) end
        ),
        SimpleRunner("simple_gpu", (ns, bs) -> (
            (manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager gpu_malloc())),
            quote (ns, bs) -> (CUDA.SimpleAreaManager(256, 100), ) end
        ),
        SimpleRunner("simple_gpu_gather", (ns, bs) -> (
            (manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager shmem=2*ns*sizeof(Int64) gpu_malloc_gather())),
            quote (ns, bs) -> (CUDA.SimpleAreaManager(256, 100), ) end
        ),
        SimpleRunner("cuda_simple", (ns, bs) -> (
            (manager,) -> (@sync @cuda threads=ns blocks=bs manager=manager policy_type=CUDA.NoPolicy cuda_malloc())),
            ((ns, bs) -> ns * bs <= 1024),
            quote (ns, bs) -> (CUDA.WarpAreaManager(8, 100), ) end,
        ),
        ExecRunner("cuda", "cuda_malloc"),
    ]
    open(file, "w") do io
        perform_benchmarks([128], [1, 8], runners ; io = io, samples=10, evals=1, measures=[:median, :std])
    end
end


function malloc_count_test(file=nothing)
    if file === nothing
        file = "tmp/malloc_count.csv"
    end

    open(file, "w") do io
        println(io, "\"chunk_size\", \"chunk_count\", \"total_count\"")
        for chunk in [1, 128, 256, 1024, 2048, 1024 * 8, 1024 * 16, 1024 * 32, 1024 * 64,1024 * 128, 1024 * 256, 1024 * 512, 1024 * 1024]
            t = exec_cmd("max_count", chunk; target="total_size") # ns to seconds
            count = div(t, chunk)
            println(io, "$chunk, $count, $t")
        end
    end
end
