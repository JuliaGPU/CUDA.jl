# warp- and block-level reductions, used by kernels in the library packages

# Reduce a value across a warp
@inline function reduce_warp(op, val)
    assume(warpsize() == 32)
    offset = 0x00000001
    while offset < warpsize()
        val = op(val, shfl_down_sync(0xffffffff, val, offset))
        offset <<= 1
    end

    return val
end

# Reduce a value across a block, using shared memory for communication
@inline function reduce_block(op, val::T, neutral, shuffle::Val{true}) where T
    # shared mem for partial sums
    assume(warpsize() == 32)
    shared = CuStaticSharedArray(T, 32)

    wid, lane = fldmod1(threadIdx().x, warpsize())

    # each warp performs partial reduction
    val = reduce_warp(op, val)

    # write reduced value to shared memory
    if lane == 1
        @inbounds shared[wid] = val
    end

    # wait for all partial reductions
    sync_threads()

    # read from shared memory only if that warp existed
    val = if threadIdx().x <= fld1(blockDim().x, warpsize())
         @inbounds shared[lane]
    else
        neutral
    end

    # final reduce within first warp
    if wid == 1
        val = reduce_warp(op, val)
    end

    return val
end
@inline function reduce_block(op, val::T, neutral, shuffle::Val{false}) where T
    threads = blockDim().x
    thread = threadIdx().x

    # The caller reserves blockDim().x * sizeof(T) bytes of shared memory.
    shared = @inbounds CuDynamicSharedArray(T, (threads,))
    @inbounds shared[thread] = val

    # perform a reduction
    d = 1
    while d < threads
        sync_threads()
        index = 2 * d * (thread-1) + 1
        @inbounds if index <= threads
            other_val = if index + d <= threads
                shared[index+d]
            else
                neutral
            end
            shared[index] = op(shared[index], other_val)
        end
        d *= 2
    end

    # load the final value on the first thread
    if thread == 1
        val = @inbounds shared[thread]
    end

    return val
end
