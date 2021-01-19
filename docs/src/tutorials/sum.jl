# # Custom sum
#
# This tutorial shows, how to implement custom reduction algorithms on CPU. Our example
# will be a custom sum. We will start with the most simple possible implementation and
# provide faster implementations later.

using CUDA

function sum_baseline(out, arr)
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
arr = CUDA.randn(10)
@cuda sum_baseline(out, arr)
@test CUDA.sum(arr) ≈ out[]

# Since we want to test and benchmark multiple variants of `sum` kernels,
# it makes sense to put the kernel launch in a function:
function run(kernel, arr)
    out = CUDA.zeros(eltype(arr))
    CUDA.@sync begin
        @cuda threads=128 blocks=1024 kernel(out, arr)
    end
    out[]
end

using BenchmarkTools
arr = CUDA.rand(10^6)
@test run(sum_baseline, arr) ≈ CUDA.sum(arr)
@btime run(sum_baseline, arr) # 1.691 ms (46 allocations: 1.41 KiB)
@btime CUDA.sum(arr) # 37.511 μs (62 allocations: 1.62 KiB)


# Our kernel is much slower, then the optimized implementation `CUDA.sum`.
# The problem is, that at the end of the day we are computing the sum sequentially.
# While we use lots of threads, only one can execute the incrementation
# `out[] += arr[i]`
# at a time.
# We can improve the situation by accessing `out[]` less often:
function sum_atomic(out, arr)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    acc = zero(eltype(out))
    for i = index:stride:length(arr)
        @inbounds acc += arr[i]
    end
    @atomic out[] += acc
    return nothing
end

arr = CUDA.rand(10^6)
@test CUDA.sum(arr) ≈ run(sum_atomic, arr)
@btime run(sum_atomic, arr) # 266.052 μs (46 allocations: 1.41 KiB)
@btime sum(arr) # 37.511 μs (62 allocations: 1.62 KiB)

# Performance is better now, but there is still a lot of threads, that want to access
# `out[]` at the same time.
