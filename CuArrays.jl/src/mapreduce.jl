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

Base.@propagate_inbounds _map_getindex(args::Tuple, I) = ((args[1][I]), _map_getindex(Base.tail(args), I)...)
Base.@propagate_inbounds _map_getindex(args::Tuple{Any}, I) = ((args[1][I]),)
Base.@propagate_inbounds _map_getindex(args::Tuple{}, I) = ()

# Reduce an array across the grid. All elements to be processed can be addressed by the
# product of the two iterators `Rreduce` and `Rother`, where the latter iterator will have
# singleton entries for the dimensions that should be reduced (and vice versa).
function partial_mapreduce_grid(f, op, neutral, Rreduce, Rother, shuffle, R, As...)
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

## COV_EXCL_STOP

if VERSION < v"1.5.0-DEV.748"
    Base.axes(bc::Base.Broadcast.Broadcasted{<:CuArrayStyle, <:NTuple{N}},
              d::Integer) where N =
        d <= N ? axes(bc)[d] : Base.OneTo(1)
end

NVTX.@range function GPUArrays.mapreducedim!(f, op, R::CuArray{T},
                                             A::Union{AbstractArray,Broadcast.Broadcasted};
                                             init=nothing) where T
    Base.check_reducedims(R, A)
    length(A) == 0 && return R # isempty(::Broadcasted) iterates

    f = cufunc(f)
    op = cufunc(op)

    # be conservative about using shuffle instructions
    shuffle = true
    shuffle &= capability(device()) >= v"3.0"
    shuffle &= T in (Bool, Int32, Int64, Float32, Float64, ComplexF32, ComplexF64)

    # add singleton dimensions to the output container, if needed
    if ndims(R) < ndims(A)
        R = reshape(R, ntuple(i -> ifelse(i <= ndims(R), size(R,i), 1), ndims(A)))
    end

    # iteration domain, split in two: one part covers the dimensions that should
    # be reduced, and the other covers the rest. combining both covers all values.
    Rall = CartesianIndices(axes(A))
    Rother = CartesianIndices(axes(R))
    Rreduce = CartesianIndices(ifelse.(axes(A) .== axes(R), Ref(Base.OneTo(1)), axes(A)))
    # NOTE: we hard-code `OneTo` (`first.(axes(A))` would work too) or we get a
    #       CartesianIndices object with UnitRanges that behave badly on the GPU.
    @assert length(Rall) == length(Rother) * length(Rreduce)

    # allocate an additional, empty dimension to write the reduced value to.
    # this does not affect the actual location in memory of the final values,
    # but allows us to write a generalized kernel supporting partial reductions.
    R′ = reshape(R, (size(R)..., 1))

    # how many threads do we want?
    #
    # threads in a block work together to reduce values across the reduction dimensions;
    # we want as many as possible to improve algorithm efficiency and execution occupancy.
    dev = device()
    wanted_threads = shuffle ? nextwarp(dev, length(Rreduce)) : nextpow(2, length(Rreduce))
    function compute_threads(max_threads)
        if wanted_threads > max_threads
            shuffle ? prevwarp(dev, max_threads) : prevpow(2, max_threads)
        else
            wanted_threads
        end
    end

    # how many threads can we launch?
    #
    # we might not be able to launch all those threads to reduce each slice in one go.
    # that's why each threads also loops across their inputs, processing multiple values
    # so that we can span the entire reduction dimension using a single thread block.
    args = (f, op, init, Rreduce, Rother, Val(shuffle), R′, A)
    kernel_args = cudaconvert.(args)
    kernel_tt = Tuple{Core.Typeof.(kernel_args)...}
    kernel = cufunction(partial_mapreduce_grid, kernel_tt)
    compute_shmem(threads) = shuffle ? 0 : 2*threads*sizeof(T)
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
        @cuda threads=threads blocks=blocks shmem=shmem partial_mapreduce_grid(
            f, op, init, Rreduce, Rother, Val(shuffle), R′, A)
    else
        # we need multiple steps to cover all values to reduce
        partial = similar(R, (size(R)..., reduce_blocks))
        if init === nothing
            # without an explicit initializer we need to copy from the output container
            sz = prod(size(R))
            for i in 1:reduce_blocks
                # TODO: async copies (or async fill!, but then we'd need to load first)
                #       or maybe just broadcast since that extends singleton dimensions
                copyto!(partial, (i-1)*sz+1, R, 1, sz)
            end
        end
        @cuda threads=threads blocks=blocks shmem=shmem partial_mapreduce_grid(
            f, op, init, Rreduce, Rother, Val(shuffle), partial, A)

        GPUArrays.mapreducedim!(identity, op, R′, partial; init=init)
    end

    return R
end
