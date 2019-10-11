# TODO: features
# - init::Some
# - pairwise
# - promote_op, and correctly handling e.g. cumsum(Bool[]) (without first converting to Int)

# TODO: performance
# - shuffle
# - warp-aggregate atomics

# partial scan of individual thread blocks within a grid
# work-efficient implementation after Blelloch (1990)
#
# number of threads needs to be a power-of-2
function partial_scan(op::Function, input::CuDeviceArray{T,N}, output::CuDeviceArray{T,N},
                      Rdim, Rpre, Rpost, Rother,
                      ::Val{inclusive}=Val(true)) where {T, N, inclusive}
    threads = blockDim().x
    thread = threadIdx().x
    block = blockIdx().x

    temp = @cuDynamicSharedMem(T, (2*threads,))

    # iterate the main dimension using threads and the first block dimension
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    # iterate the other dimensions using the remaining block dimensions
    j = (blockIdx().z-1) * gridDim().y + blockIdx().y

    if j > length(Rother)
        return
    end

    I = Rother[j]
    Ipre = Rpre[I[1]]
    Ipost = Rpost[I[2]]

    # load input into shared memory
    @inbounds temp[thread] = if i <= length(Rdim)
        input[Ipre, i, Ipost]
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
    @inbounds if i <= length(Rdim)
        output[Ipre, i, Ipost] = if inclusive
            op(temp[thread], input[Ipre, i, Ipost])
        else
            temp[thread]
        end
    end

    return
end

# aggregate the result of a partial scan by applying preceding block aggregates
function aggregate_partial_scan(op::Function, output::CuDeviceArray{T,N},
                                aggregates::CuDeviceArray{T,N}, Rdim, Rpre, Rpost, Rother
                               ) where {T,N}
    threads = blockDim().x
    thread = threadIdx().x
    block = blockIdx().x

    # iterate the main dimension using threads and the first block dimension
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    # iterate the other dimensions using the remaining block dimensions
    j = (blockIdx().z-1) * gridDim().y + blockIdx().y

    @inbounds if block > 1 && i <= length(Rdim) && j <= length(Rother)
        I = Rother[j]
        Ipre = Rpre[I[1]]
        Ipost = Rpost[I[2]]

        output[Ipre, i, Ipost] = op(aggregates[Ipre, block-1, Ipost], output[Ipre, i, Ipost])
    end

    return
end

function Base._accumulate!(f::Function, output::CuArray{T,N}, input::CuArray{T,N},
                           dims::Integer, init::Nothing) where {T,N}
    dims > 0 || throw(ArgumentError("dims must be a positive integer"))
    inds_t = axes(input)
    axes(output) == inds_t || throw(DimensionMismatch("shape of B must match A"))
    dims > ndims(input) && return copyto!(output, input)
    isempty(inds_t[dims]) && return output

    # iteration domain across the main dimension
    Rdim = CartesianIndices((size(input, dims),))

    # iteration domain for the other dimensions
    Rpre = CartesianIndices(size(input)[1:dims-1])
    Rpost = CartesianIndices(size(input)[dims+1:end])
    Rother = CartesianIndices((length(Rpre), length(Rpost)))

    # determine how many threads we can launch for the scan kernel
    args = (f, input, output, Rdim, Rpre, Rpost, Rother)
    kernel_args = cudaconvert.(args)
    kernel_tt = Tuple{Core.Typeof.(kernel_args)...}
    kernel = cufunction(partial_scan, kernel_tt)
    kernel_config = launch_configuration(kernel.fun; shmem=(threads)->2*threads*sizeof(T))

    # determine the grid layout to cover the other dimensions
    dev = CUDAdrv.device(kernel.fun.mod.ctx)
    max_other_blocks = attribute(dev, CUDAdrv.MAX_GRID_DIM_Y)
    blocks_other = (min(length(Rother), max_other_blocks),
                    cld(length(Rother), max_other_blocks))

    # does that suffice to scan the array in one go?
    full = nextpow(2, length(Rdim))
    if full <= kernel_config.threads
        @cuda(threads=full, blocks=(1, blocks_other...), shmem=2*full*sizeof(T), name="scan",
              partial_scan(f, input, output, Rdim, Rpre, Rpost, Rother))
    else
        # perform partial scans across the scanning dimension
        partial = prevpow(2, kernel_config.threads)
        blocks_dim = cld(length(Rdim), partial)
        @cuda(threads=partial, blocks=(blocks_dim, blocks_other...), shmem=2*partial*sizeof(T),
              partial_scan(f, input, output, Rdim, Rpre, Rpost, Rother))

        # get the total of each thread block (except the first) of the partial scans
        aggregates = zeros(T, Base.setindex(size(input), blocks_dim, dims))
        copyto!(aggregates, selectdim(output, dims, partial:partial:length(Rdim)))

        # scan these totals to get totals for the entire partial scan
        accumulate!(f, aggregates, aggregates; dims=dims)

        # add those totals to the partial scan result
        # NOTE: we assume that this kernel requires fewer resources than the scan kernel.
        #       if that does not hold, launch with fewer threads and calculate
        #       the aggregate block index within the kernel itself.
        @cuda(threads=partial, blocks=(blocks_dim, blocks_other...),
              aggregate_partial_scan(f, output, aggregates, Rdim, Rpre, Rpost, Rother))

        CuArrays.unsafe_free!(aggregates)
    end

    return output
end

function Base._accumulate!(op::Function, vout::CuArray{T}, v::CuArray, dims,
                           init) where {T}
    vin = T.(v)  # convert to vector with eltype T

    return Base._accumulate!(op::Function, vout, vin, dims, init)
end

function Base._accumulate!(op::Function, vout::CuVector{T}, v::CuVector{T}, dims::Nothing,
                           init::Nothing) where {T}
    return Base._accumulate!(op::Function, vout, v, 1, init)
end

function Base._accumulate!(op::Function, vout::CuVector{T}, v::CuVector, dims::Nothing,
                           init::Nothing) where {T}
    vin = T.(v)  # convert to vector with eltype T

    return Base._accumulate!(op::Function, vout, vin, 1, init)
end

Base.accumulate_pairwise!(op, result::CuVector, v::CuVector) = accumulate!(op, result, v)
