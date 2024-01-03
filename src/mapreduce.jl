## COV_EXCL_START

# TODO
# - serial version for lower latency
# - block-stride loop to delay need for second kernel launch

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

    # shared mem for a complete reduction
    shared = CuDynamicSharedArray(T, (threads,))
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

Base.@propagate_inbounds _map_getindex(args::Tuple, I) = ((args[1][I]), _map_getindex(Base.tail(args), I)...)
Base.@propagate_inbounds _map_getindex(args::Tuple{Any}, I) = ((args[1][I]),)
Base.@propagate_inbounds _map_getindex(args::Tuple{}, I) = ()

# Reduce an array across the grid. All elements to be processed can be addressed by the
# product of the two iterators `Rreduce` and `Rother`, where the latter iterator will have
# singleton entries for the dimensions that should be reduced (and vice versa).
function partial_mapreduce_grid(f, op, neutral, Rreduce, Rother, shuffle, R, As...)
    assume(length(Rother) > 0)

    # decompose the 1D hardware indices into separate ones for reduction (across threads
    # and possibly blocks if it doesn't fit) and other elements (remaining blocks)
    threadIdx_reduce = threadIdx().x
    blockDim_reduce = blockDim().x
    blockIdx_reduce, blockIdx_other = fldmod1(blockIdx().x, length(Rother))
    gridDim_reduce = gridDim().x ÷ length(Rother)

    # block-based indexing into the values outside of the reduction dimension
    # (that means we can safely synchronize threads within this block)
    iother = blockIdx_other
    @inbounds if iother <= length(Rother)
        Iother = Rother[iother]

        # load the neutral value
        Iout = CartesianIndex(Tuple(Iother)..., blockIdx_reduce)
        neutral = if neutral === nothing
            R[Iout]
        else
            neutral
        end

        val = op(neutral, neutral)

        # reduce serially across chunks of input vector that don't fit in a block
        ireduce = threadIdx_reduce + (blockIdx_reduce - 1) * blockDim_reduce
        while ireduce <= length(Rreduce)
            Ireduce = Rreduce[ireduce]
            J = max(Iother, Ireduce)
            val = op(val, f(_map_getindex(As, J)...))
            ireduce += blockDim_reduce * gridDim_reduce
        end

        val = reduce_block(op, val, neutral, shuffle)

        # write back to memory
        if threadIdx_reduce == 1
            R[Iout] = val
        end
    end

    return
end

function big_mapreduce_kernel(f, op, neutral, Rreduce, Rother, R, As)
    grid_idx = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    @inbounds if grid_idx <= length(Rother)
        Iother = Rother[grid_idx]

        # load the neutral value
        neutral = if neutral === nothing
            R[Iother]
        else
            neutral
        end

        val = op(neutral, neutral)

        Ibegin = Rreduce[1]
        for Ireduce in Rreduce
            val = op(val, f(As[Iother + Ireduce - Ibegin]))
        end
        R[Iother] = val
    end
    return
end

## COV_EXCL_STOP

# factored out for use in tests
function big_mapreduce_threshold(dev)
    max_concurrency = attribute(dev, DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK) *
                      attribute(dev, DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    return max_concurrency
end

function GPUArrays.mapreducedim!(f::F, op::OP, R::AnyCuArray{T},
                                 A::Union{AbstractArray,Broadcast.Broadcasted};
                                 init=nothing) where {F, OP, T}
    Base.check_reducedims(R, A)
    length(A) == 0 && return R # isempty(::Broadcasted) iterates
    dev = device()

    # be conservative about using shuffle instructions
    shuffle = T <: Union{Bool,
                         UInt8, UInt16, UInt32, UInt64, UInt128,
                         Int8, Int16, Int32, Int64, Int128,
                         Float16, Float32, Float64,
                         ComplexF16, ComplexF32, ComplexF64}

    # add singleton dimensions to the output container, if needed
    if ndims(R) < ndims(A)
        dims = Base.fill_to_length(size(R), 1, Val(ndims(A)))
        R = reshape(R, dims)
    end

    # iteration domain, split in two: one part covers the dimensions that should
    # be reduced, and the other covers the rest. combining both covers all values.
    Rall = CartesianIndices(axes(A))
    Rother = CartesianIndices(axes(R))
    Rreduce = CartesianIndices(ifelse.(axes(A) .== axes(R), Ref(Base.OneTo(1)), axes(A)))
    # NOTE: we hard-code `OneTo` (`first.(axes(A))` would work too) or we get a
    #       CartesianIndices object with UnitRanges that behave badly on the GPU.
    @assert length(Rall) == length(Rother) * length(Rreduce)
    @assert length(Rother) > 0

    # If `Rother` is large enough, then a naive loop is more efficient than partial reductions.
    if length(Rother) >= big_mapreduce_threshold(dev)
        args = (f, op, init, Rreduce, Rother, R, A)
        kernel = @cuda launch=false big_mapreduce_kernel(args...)
        kernel_config = launch_configuration(kernel.fun)
        threads = kernel_config.threads
        blocks = cld(length(Rother), threads)
        kernel(args...; threads, blocks)
        return R
    end

    # allocate an additional, empty dimension to write the reduced value to.
    # this does not affect the actual location in memory of the final values,
    # but allows us to write a generalized kernel supporting partial reductions.
    R′ = reshape(R, (size(R)..., 1))

    # how many threads do we want?
    #
    # threads in a block work together to reduce values across the reduction dimensions;
    # we want as many as possible to improve algorithm efficiency and execution occupancy.

    wanted_threads = shuffle ? nextwarp(dev, length(Rreduce)) : length(Rreduce)
    function compute_threads(max_threads)
        if wanted_threads > max_threads
            shuffle ? prevwarp(dev, max_threads) : max_threads
        else
            wanted_threads
        end
    end

    # how many threads can we launch?
    #
    # we might not be able to launch all those threads to reduce each slice in one go.
    # that's why each threads also loops across their inputs, processing multiple values
    # so that we can span the entire reduction dimension using a single thread block.
    kernel = @cuda launch=false partial_mapreduce_grid(f, op, init, Rreduce, Rother, Val(shuffle), R′, A)
    compute_shmem(threads) = shuffle ? 0 : threads*sizeof(T)
    kernel_config = launch_configuration(kernel.fun; shmem=compute_shmem∘compute_threads)
    reduce_threads = compute_threads(kernel_config.threads)
    reduce_shmem = compute_shmem(reduce_threads)

    # how many blocks should we launch?
    #
    # even though we can always reduce each slice in a single thread block, that may not be
    # optimal as it might not saturate the GPU. we already launch some blocks to process
    # independent dimensions in parallel; pad that number to ensure full occupancy.
    other_blocks = length(Rother)
    reduce_blocks = if other_blocks >= kernel_config.blocks
        1
    else
        min(cld(length(Rreduce), reduce_threads),       # how many we need at most
            cld(kernel_config.blocks, other_blocks))    # maximize occupancy
    end

    # determine the launch configuration
    threads = reduce_threads
    shmem = reduce_shmem
    blocks = reduce_blocks*other_blocks

    # perform the actual reduction
    if reduce_blocks == 1
        # we can cover the dimensions to reduce using a single block
        kernel(f, op, init, Rreduce, Rother, Val(shuffle), R′, A; threads, blocks, shmem)
    else
        # we need multiple steps to cover all values to reduce
        partial = similar(R, (size(R)..., reduce_blocks))
        if init === nothing
            # without an explicit initializer we need to copy from the output container
            partial .= R
        end
        # NOTE: we can't use the previously-compiled kernel, since the type of `partial`
        #       might not match the original output container (e.g. if that was a view).
        @cuda(threads, blocks, shmem,
              partial_mapreduce_grid(f, op, init, Rreduce, Rother, Val(shuffle), partial, A))

        GPUArrays.mapreducedim!(identity, op, R′, partial; init=init)
    end

    return R
end
