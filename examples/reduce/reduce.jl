# EXCLUDE FROM TESTING
# this file doesn't have an entry point, see `verify.jl` instead

# Fast parallel reduction for Kepler hardware
# - uses shuffle and shared memory to reduce efficiently
# - support for large arrays
#
# Based on devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/

using CUDAdrv, CUDAnative

include(joinpath(@__DIR__, "..", "..", "test", "array.jl"))
const CuArray = CuTestArray    # real applications: use CuArrays.jl


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

    return
end

"""
Reduce a large array.

Kepler-specific implementation, ie. you need sm_30 or higher to run this code.
"""
function gpu_reduce(op::Function, input::CuArray{T}, output::CuArray{T}) where {T}
    len = length(input)

    function get_config(kernel)
        fun = kernel.fun
        config = launch_configuration(fun)

        # pad to cover as many elements as possible (`cld`), but keep in mind
        # how we launch this kernel for the final reduction step `min(..., threads)`
        blocks = min(cld(len, config.threads), config.threads)

        # the output array must have a size equal to or larger than the number of blocks
        # in the grid because each block writes to a unique location within the array.
        if length(output) < blocks
            throw(ArgumentError("output array too small, should be at least $blocks elements"))
        end

        return (threads=config.threads, blocks=blocks)
    end

    # NOTE: manual expansion of @cuda because we need the result of the dynamic launch config
    #@cuda config=get_config reduce_grid(op, input, output, len)
    args = (op, input, output, len)
    GC.@preserve args begin
        kernel_args = cudaconvert.(args)
        kernel_tt = Tuple{Core.Typeof.(kernel_args)...}
        kernel = cufunction(reduce_grid, kernel_tt)
        kernel_config = get_config(kernel)
        kernel(kernel_args...; kernel_config...)
    end

    @cuda threads=kernel_config.threads reduce_grid(op, output, output, kernel_config.blocks)
end


# FURTHER IMPROVEMENTS:
# - use atomic memory operations
# - dynamic block/grid size based on device capabilities
# - vectorized memory access
#   devblogs.nvidia.com/parallelforall/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
