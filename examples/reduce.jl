using CUDAdrv, CUDAnative
using Base.Test

# Fast parallel reduction for Kepler hardware
#
# Based on devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
#
# FURTHER IMPROVEMENTS:
# - analyze LLVM IR for redundant Int32/Int64 conversions
#   without adding Int32() everywhere (JuliaGPU/CUDAnative.jl#25)
# - use atomic memory operations
# - add dispatch-based fallbacks for non-Kepler hardware
# - dynamic block/grid size based on device capabilities
# - improve documentation
# - vectorized memory access
#   devblogs.nvidia.com/parallelforall/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

# Reduce a value across a warp
function reduce_warp{F<:Function,T}(op::F, val::T)::T
    offset = CUDAnative.warpsize() ÷ Int32(2)
    # TODO: this can be unrolled if warpsize is known...
    while offset > Int32(0)
        val = op(val, shfl_down(val, offset))
        offset ÷= Int32(2)
    end
    return val
end

# Reduce a value across a block, using shared memory for communication
function reduce_block{F<:Function,T}(op::F, val::T)::T
    # shared mem for 32 partial sums
    shared = @cuStaticSharedMem(T, 32)

    wid, lane = fldmod1(threadIdx().x, CUDAnative.warpsize())

    # each warp performs partial reduction
    val = reduce_warp(op, val)

    # write reduced value to shared memory
    if lane == 1
        @inbounds shared[wid] = val
    end

    # wait for all partial reductions
    sync_threads()

    # read from shared memory only if that warp existed
    @inbounds val = (threadIdx().x <= fld(blockDim().x, CUDAnative.warpsize())) ? shared[lane] : zero(T)

    # final reduce within first warp
    if wid == 1
        val = reduce_warp(op, val)
    end

    return val
end

# Reduce an array across a complete grid
function reduce_grid{F<:Function,T}(op::F, input::CuDeviceArray{T,1}, output::CuDeviceArray{T,1}, N::Integer)
    # TODO: neutral element depends on the operator (see Base's 2 and 3 argument `reduce`)
    val = zero(T)

    # reduce multiple elements per thread (grid-stride loop)
    # TODO: step range (see JuliaGPU/CUDAnative.jl#12)
    i = (blockIdx().x-Int32(1)) * blockDim().x + threadIdx().x
    step = blockDim().x * gridDim().x
    while i <= N
        @inbounds val = op(val, input[i])
        i += step
    end

    val = reduce_block(op, val)

    if threadIdx().x == Int32(1)
        @inbounds output[blockIdx().x] = val
    end

    return
end

"""
Reduce a large array.

Kepler-specific implementation, ie. you need sm_30 or higher to run this code.
"""
function gpu_reduce{F<:Function,T}(op::F, input::CuArray{T,1}, output::CuArray{T,1})
    ctx = CuCurrentContext()
    dev = device(ctx)
    @assert(capability(dev) >= v"3.0", "this implementation requires a newer GPU")
    N = length(input)

    # TODO: these values are hardware-dependent, with recent GPUs supporting more threads
    threads = 512
    blocks = min((N + threads - 1) ÷ threads, 1024)

    # the output array must have a size equal to or larger than the number of thread blocks
    # in the grid because each block writes to a unique location within the array.
    if length(output) < blocks
        throw(ArgumentError("output array too small, should be at least $blocks elements"))
    end

    @cuda (blocks,threads) reduce_grid(op, input, output, N)
    @cuda (1,1024) reduce_grid(op, output, output, blocks)

    return
end


dev = CuDevice(0)
ctx = CuContext(dev)

len = 123456

a = ones(Int32,len)

cpu_a = copy(a)
cpu_a = reduce(+, cpu_a)

gpu_a = CuArray(a)
gpu_b = similar(gpu_a)
gpu_reduce(+, gpu_a, gpu_b)

@assert cpu_a ≈ Array(gpu_b)[1]

destroy(ctx)
