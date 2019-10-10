# TODO: features
# - init::Some
# - CuMatrix
# - pairwise

# TODO: performance
# - shuffle
# - warp-aggregate atomics

# partial scan of individual thread blocks within a grid
# work-efficient implementation after Blelloch (1990)
#
# number of threads needs to be a power-of-2
function partial_scan(op::Function, input::CuDeviceVector{T}, output::CuDeviceVector{T},
                      ::Val{inclusive}=Val(true)) where {T, inclusive}
    threads = blockDim().x
    thread = threadIdx().x
    block = blockIdx().x

    temp = @cuDynamicSharedMem(T, (2*threads,))

    i = (block-1) * threads + thread

    # load input into shared memory
    @inbounds temp[thread] = if i <= length(input)
        input[i]
    else
        zero(T)
    end

    # build sum in place up the tree
    offset = 1
    d = threads>>1
    while d > 0
        sync_threads()
        @inbounds if thread <= d
            ai = offset * (2*thread-1)
            bi = offset * (2*thread)
            temp[bi] = op(temp[ai], temp[bi])
        end
        offset *= 2
        d >>= 1
    end

    # clear the last element
    @inbounds if thread == 1
        temp[threads] = 0
    end

    # traverse down tree & build scan
    d = 1
    while d < threads
        offset >>= 1
        sync_threads()
        @inbounds if thread <= d
            ai = offset * (2*thread-1)
            bi = offset * (2*thread)

            t = temp[ai]
            temp[ai] = temp[bi]
            temp[bi] = op(t, temp[bi])
        end
        d *= 2
    end

    sync_threads()

    # write results to device memory
    @inbounds if i <= length(input)
        output[i] = if inclusive
            op(temp[thread], input[i])
        else
            temp[thread]
        end
    end

    return
end

# aggregate the result of a partial scan by applying preceding block aggregates
function aggregate_partial_scan(op::Function, output::CuDeviceVector{T},
                                aggregates::CuDeviceVector{T}) where {T}
    threads = blockDim().x
    thread = threadIdx().x
    block = blockIdx().x

    i = (block-1) * threads + thread

    @inbounds if block > 1 && i <= length(output)
        output[i] = op(aggregates[block-1], output[i])
    end

    return
end

function Base._accumulate!(f::Function, output::CuVector{T}, input::CuVector{T},
                           dims::Nothing, init::Nothing) where {T}
    length(input) == 0 && return output

    # determine how many threads we can launch for the scan kernel
    args = (+, input, output)
    kernel_args = cudaconvert.(args)
    kernel_tt = Tuple{Core.Typeof.(kernel_args)...}
    kernel = cufunction(partial_scan, kernel_tt)
    kernel_config = launch_configuration(kernel.fun; shmem=(threads)->2*threads*sizeof(T))

    # does that suffice to scan the array in one go?
    full = nextpow(2, length(input))
    if full <= kernel_config.threads
        @cuda name="scan" threads=full shmem=2*full*sizeof(T) partial_scan(+, input, output)
    else
        # perform partial scans of smaller thread blocks
        partial = prevpow(2, kernel_config.threads)
        blocks = cld(length(input), partial)
        @cuda threads=partial blocks=blocks shmem=2*partial*sizeof(T) partial_scan(f, input, output)

        # calculate per-block totals
        aggregates = CuArrays.zeros(T, nextpow(2, blocks))
        copyto!(aggregates, output[partial:partial:end])

        # scan block totals to get block aggregates
        accumulate!(f, aggregates, aggregates)

        # apply the block aggregates to the partial scan result
        # NOTE: we assume that this kernel requires fewer resources than the scan kernel.
        #       if that does not hold, launch with fewer threads and calculate
        #       the aggregate block index within the kernel itself.
        @cuda threads=partial blocks=blocks aggregate_partial_scan(+, output, aggregates)

        CuArrays.unsafe_free!(aggregates)
    end

    return output
end

function Base._accumulate!(op::Function, vout::CuVector{T}, v::CuVector, dims::Int,
                           init::Nothing) where {T}
    if dims != 1
        return copyto!(vout, v)
    end

    return Base._accumulate!(op::Function, vout::CuVector{T}, v::CuVector, nothing, nothing)
end

function Base._accumulate!(op::Function, vout::CuVector{T}, v::CuVector, dims::Nothing,
                           init::Nothing) where {T}
    vin = T.(v)  # convert to vector with eltype T

    return Base._accumulate!(op::Function, vout::CuVector{T}, vin::CuVector, dims, init)
end

Base.accumulate_pairwise!(op, result::CuVector, v::CuVector) = accumulate!(op, result, v)
