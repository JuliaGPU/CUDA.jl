# EXCLUDE FROM TESTING
# this file doesn't have an entry point, see `verify.jl` instead

# Fast parallel reduction for Kepler hardware
# - uses shuffle and shared memory to reduce efficiently
# - support for large arrays
#
# Based on devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/

using CUDAdrv, CUDAnative


#
# Main implementation
#

# Reduce a value across a warp
@inline function reduce_warp(op::F, val::T)::T where {F<:Function,T}
    offset = CUDAnative.warpsize() ÷ 2
    # TODO: this can be unrolled if warpsize is known...
    while offset > 0
        val = op(val, shfl_down(val, offset))
        offset ÷= 2
    end
    return val
end

# Reduce a value across a block, using shared memory for communication
@inline function reduce_block(op::F, val::T)::T where {F<:Function,T}
    # shared mem for 32 partial sums
    shared = @cuStaticSharedMem(T, 32)

    # TODO: use fldmod1 (JuliaGPU/CUDAnative.jl#28)
    wid  = div(threadIdx().x-1, CUDAnative.warpsize()) + 1
    lane = rem(threadIdx().x-1, CUDAnative.warpsize()) + 1

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
function reduce_grid(op::F, input::CuDeviceVector{T}, output::CuDeviceVector{T},
                     len::Integer) where {F<:Function,T}

    # TODO: neutral element depends on the operator (see Base's 2 and 3 argument `reduce`)
    val = zero(T)

    # reduce multiple elements per thread (grid-stride loop)
    # TODO: step range (see JuliaGPU/CUDAnative.jl#12)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    step = blockDim().x * gridDim().x
    while i <= len
        @inbounds val = op(val, input[i])
        i += step
    end

    val = reduce_block(op, val)

    if threadIdx().x == 1
        @inbounds output[blockIdx().x] = val
    end
end

"""
Reduce a large array.

Kepler-specific implementation, ie. you need sm_30 or higher to run this code.
"""
function gpu_reduce(op::Function, input::CuVector{T}, output::CuVector{T}) where {T}
    len = length(input)

    # TODO: these values are hardware-dependent, with recent GPUs supporting more threads
    threads = 512
    blocks = min((len + threads - 1) ÷ threads, 1024)

    # the output array must have a size equal to or larger than the number of thread blocks
    # in the grid because each block writes to a unique location within the array.
    if length(output) < blocks
        throw(ArgumentError("output array too small, should be at least $blocks elements"))
    end

    @cuda blocks=blocks threads=threads reduce_grid(op, input, output, len)
    @cuda threads=1024 reduce_grid(op, output, output, blocks)
end


# FURTHER IMPROVEMENTS:
# - use atomic memory operations
# - dynamic block/grid size based on device capabilities
# - vectorized memory access
#   devblogs.nvidia.com/parallelforall/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
