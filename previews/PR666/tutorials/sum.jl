# # Custom sum
#
# This tutorial shows, how to implement custom reduction algorithms on CPU. Our example
# will be a custom sum. We will start with the most simple possible implementation and
# provide faster implementations later.

using CUDA

function sum_baseline_kernel(out, arr)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i = index:stride:length(arr)
        @atomic out[] += arr[i]
    end
    return nothing
end

# Observe that kernels have no return values. So instead of returning the result,
# we write it in a 0-dimensional array `out`.
# First we should check, whether our kernel is correct
using Test
out = CUDA.zeros()
arr = CUDA.randn(10^6)
@cuda sum_baseline_kernel(out, arr)
@test CUDA.sum(arr) ≈ out[]

# Lets wrap the kernel in a function, that takes care of
# allocating `out` and launching the kernel.
function sum_baseline(arr)
    out = CUDA.zeros(eltype(arr))
    CUDA.@sync begin
        @cuda threads=256 blocks=64 sum_baseline_kernel(out, arr)
    end
    out[]
end

using BenchmarkTools
@test sum_baseline(arr) ≈ CUDA.sum(arr)
@btime sum_baseline(arr) # 1.675 ms (9 allocations: 368 bytes)
@btime CUDA.sum(arr) # 42.621 μs (32 allocations: 1.25 KiB)

# Our kernel is much slower, then the optimized implementation `CUDA.sum`.
# The problem is, that at the end of the day we are computing the sum sequentially.
# While we use lots of threads, only one can execute the incrementation
# `out[] += arr[i]`
# at a time.
# We can improve the situation by accessing `out[]` less often:
function sum_atomic_kernel(out, arr)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    acc = zero(eltype(out))
    for i = index:stride:length(arr)
        @inbounds acc += arr[i]
    end
    @atomic out[] += acc
    return nothing
end
function sum_atomic(arr)
    out = CUDA.zeros(eltype(arr))
    CUDA.@sync begin
        @cuda threads=256 blocks=64 sum_atomic_kernel(out, arr)
    end
    out[]
end

@test CUDA.sum(arr) ≈ sum_atomic(arr)
@btime sum_atomic(arr) # 61.939 μs (9 allocations: 368 bytes)
@btime sum(arr) # 42.652 μs (32 allocations: 1.25 KiB)

# Performance is much better now, but there is still a gap.
# The reason is that there is still a multitude of threads which desire access `out[]` at the same time.
#
# We can further improve the situation by using shared memory.
# The following code is based on a [CUDA tutorial](https://sodocumentation.net/cuda/topic/6566/parallel-reduction--e-g--how-to-sum-an-array-).
function sum_block_kernel!(out, x)
    ithread = threadIdx().x
    iblock = blockIdx().x
    index = (iblock - 1) * blockDim().x + ithread
    stride = blockDim().x * gridDim().x
    acc = zero(eltype(out))
    for i in index:stride:length(x)
        @inbounds acc += x[i]
    end
    shmem = @cuDynamicSharedMem(eltype(out), blockDim().x)
    shmem[ithread] = acc
    imax = blockDim().x ÷ 2
    sync_threads()
    while imax >= 1
        if ithread <= imax
            shmem[ithread] += shmem[ithread+imax]
        end
        imax = imax ÷ 2
        sync_threads()
    end
    if ithread === 1
        out[iblock] = shmem[1]
    end
    nothing
end

function sum_shmem(x)
    threads = 256
    blocks = 64
    block_sums = CUDA.zeros(eltype(x), blocks)
    CUDA.@sync begin
        shmem = sizeof(eltype(x)) * threads
        @assert ispow2(threads)
        k = @cuda threads=threads blocks=blocks shmem=shmem sum_block_kernel!(block_sums, x)
    end
    out = CUDA.zeros(eltype(x), 1)
    CUDA.@sync begin
        shmem = sizeof(eltype(x)) * threads
        @assert ispow2(threads)
        k = @cuda threads=threads blocks=1 shmem=shmem sum_block_kernel!(out, block_sums)
    end
    return only(collect(out))
end

@test CUDA.sum(arr) ≈ sum_shmem(arr)
@btime sum_shmem(arr) # 47.592 μs (16 allocations: 624 bytes)
@btime sum(arr) # 42.772 μs (32 allocations: 1.25 KiB)

# Now we are pretty close to `sum`.
