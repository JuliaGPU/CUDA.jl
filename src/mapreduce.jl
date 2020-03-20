## COV_EXCL_START

# TODO
# - serial version for lower latency
# - block-stride loop to delay need for second kernel launch

# Reduce a value across a warp
@inline function reduce_warp(op, val)
    # offset = CUDAnative.warpsize() ÷ 2
    # while offset > 0
    #     val = op(val, shfl_down_sync(0xffffffff, val, offset))
    #     offset ÷= 2
    # end

    # Loop unrolling for warpsize = 32
    val = op(val, shfl_down_sync(0xffffffff, val, 16, 32))
    val = op(val, shfl_down_sync(0xffffffff, val, 8, 32))
    val = op(val, shfl_down_sync(0xffffffff, val, 4, 32))
    val = op(val, shfl_down_sync(0xffffffff, val, 2, 32))
    val = op(val, shfl_down_sync(0xffffffff, val, 1, 32))

    return val
end

# Reduce a value across a block, using shared memory for communication
@inline function reduce_block(op, val::T, neutral, shuffle::Val{true}) where T
    # shared mem for 32 partial sums
    shared = @cuStaticSharedMem(T, 32)  # NOTE: this is an upper bound; better detect it

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
    val = if threadIdx().x <= fld1(blockDim().x, CUDAnative.warpsize())
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

    # shared mem for a complete reduction
    shared = @cuDynamicSharedMem(T, (2*threads,))
    @inbounds shared[thread] = val

    # perform a reduction
    d = threads>>1
    while d > 0
        sync_threads()
        if thread <= d
            shared[thread] = op(shared[thread], shared[thread+d])
        end
        d >>= 1
    end

    # load the final value on the first thread
    if thread == 1
        val = @inbounds shared[thread]
    end

    return val
end

# Reduce an array across the grid. All elements to be processed can be addressed by the
# product of the two iterators `Rreduce` and `Rother`, where the latter iterator will have
# singleton entries for the dimensions that should be reduced (and vice versa).
function mapreduce_grid(f, op, A, R, neutral, Rreduce, Rother, shuffle)
    # block-based indexing into the values outside of the reduction dimension
    # (that means we can safely synchronize threads within this block)
    iother = blockIdx().x
    @inbounds if iother <= length(Rother)
        Iother = Rother[iother]

        # load the neutral value
        Iout = Iother
        neutral = if neutral === nothing
            R[Iout]
        else
            neutral
        end

        val = op(neutral, neutral)

        # reduce serially across chunks of input vector that don't fit in a block
        ireduce = threadIdx().x
        while ireduce <= length(Rreduce)
            Ireduce = Rreduce[ireduce]
            J = max(Iother, Ireduce)
            val = op(val, f(A[J]))
            ireduce += blockDim().x
        end

        # reduce in parallel within the current block
        val = reduce_block(op, val, neutral, shuffle)

        # write back to memory
        if threadIdx().x == 1
            R[Iout] = val
        end
    end

    return
end

## COV_EXCL_STOP

NVTX.@range function GPUArrays.mapreducedim!(f, op, R::CuArray{T}, A::AbstractArray, init=nothing) where T
    Base.check_reducedims(R, A)
    isempty(A) && return R

    f = cufunc(f)
    op = cufunc(op)

    # be conservative about using shuffle instructions
    shuffle = true
    shuffle &= capability(device()) >= v"3.0"
    shuffle &= T in (Int32, Int64, Float32, Float64, ComplexF32, ComplexF64)
    # TODO: add support for Bool (CUDAnative.jl#420)

    # iteration domain, split in two: one part covers the dimensions that should
    # be reduced, and the other covers the rest. combining both covers all values.
    Rall = CartesianIndices(A)
    Rother = CartesianIndices(R)
    Rreduce = CartesianIndices(ifelse.(axes(A) .== axes(R), Ref(Base.OneTo(1)), axes(A)))
    # NOTE: we hard-code `OneTo` (`first.(axes(A))` would work too) or we get a
    #       CartesianIndices object with UnitRanges that behave badly on the GPU.
    @assert length(Rall) == length(Rother) * length(Rreduce)

    function configurator(kernel)
        config = launch_configuration(kernel.fun)
        dev = device()

        threads = shuffle ? nextwarp(dev, length(Rreduce)) : nextpow(2, length(Rreduce))
        if threads > config.threads
            threads = shuffle ? prevwarp(dev, config.threads) : prevpow(2, config.threads)
        end
        blocks = length(Rother)
        shmem = shuffle ? 0 : 2*threads*sizeof(T)

        return (threads=threads, blocks=blocks, shmem=shmem)
    end

    @cuda config=configurator mapreduce_grid(f, op, A, R, init, Rreduce, Rother, Val(shuffle))

    return R
end
