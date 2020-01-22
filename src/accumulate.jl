# scan and accumulate

# FIXME: certain Base operations, like accumulate, don't allow to pass a neutral element
#        since it is not required by the Base implementation (as opposed to eg. reduce).
#        to stick to the API, we use global state to provide a callback.
const neutral_elements = Dict{Function,Function}(
    Base.:(+)       => zero,
    Base.add_sum    => zero,
    Base.:(*)       => one,
    Base.mul_prod   => one,
)
function neutral_element!(op, f)
    if haskey(neutral_elements, op)
        @warn "Overriding neutral element for $op"
    end
    neutral_elements[op] = f
end
function neutral_element(op, T)
    if !haskey(neutral_elements, op)
        error("""CuArrays.jl needs to know the neutral element for your operator $op.
                 Please register your operator using: `CuArrays.neutral_element!($op, f::Function)`,
                 providing a function that returns a neutral element for a given element type.""")
    end
    f = neutral_elements[op]
    return f(T)
end

## COV_EXCL_START

# partial scan of individual thread blocks within a grid
# work-efficient implementation after Blelloch (1990)
#
# number of threads needs to be a power-of-2
#
# performance TODOs:
# - shuffle
# - warp-aggregate atomics
# - the ND case is quite a bit slower than the 1D case (not using Cartesian indices,
#   before 35fcbde1f2987023229034370b0c9091e18c4137). optimize or special-case?
function partial_scan(op::Function, output::CuDeviceArray{T}, input::CuDeviceArray,
                      Rdim, Rpre, Rpost, Rother, neutral, init,
                      ::Val{inclusive}=Val(true)) where {T, inclusive}
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

    @inbounds begin
        I = Rother[j]
        Ipre = Rpre[I[1]]
        Ipost = Rpost[I[2]]
    end

    # load input into shared memory (apply `op` to have the correct type)
    @inbounds temp[thread] = if i <= length(Rdim)
        op(neutral, input[Ipre, i, Ipost])
    else
        op(neutral, neutral)
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
        temp[threads] = neutral
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
        val = if inclusive
            op(temp[thread], input[Ipre, i, Ipost])
        else
            temp[thread]
        end
        if init !== nothing
            val = op(something(init), val)
        end
        output[Ipre, i, Ipost] = val
    end

    return
end

# aggregate the result of a partial scan by applying preceding block aggregates
function aggregate_partial_scan(op::Function, output::CuDeviceArray,
                                aggregates::CuDeviceArray, Rdim, Rpre, Rpost, Rother,
                                init)
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

        val = op(aggregates[Ipre, block-1, Ipost], output[Ipre, i, Ipost])
        if init !== nothing
            val = op(something(init), val)
        end
        output[Ipre, i, Ipost] = val
    end

    return
end

## COV_EXCL_STOP

function scan!(f::Function, output::CuArray{T}, input::CuArray;
               dims::Integer, init=nothing, neutral=neutral_element(f, T)) where {T}

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
    args = (f, output, input, Rdim, Rpre, Rpost, Rother, neutral, init, Val(true))
    kernel_args = cudaconvert.(args)
    kernel_tt = Tuple{Core.Typeof.(kernel_args)...}
    kernel = cufunction(partial_scan, kernel_tt)
    kernel_config = launch_configuration(kernel.fun; shmem=(threads)->2*threads*sizeof(T))

    # determine the grid layout to cover the other dimensions
    if length(Rother) > 1
        dev = CUDAdrv.device(kernel.fun.mod.ctx)
        max_other_blocks = attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
        blocks_other = (min(length(Rother), max_other_blocks),
                        cld(length(Rother), max_other_blocks))
    else
        blocks_other = (1, 1)
    end

    # does that suffice to scan the array in one go?
    full = nextpow(2, length(Rdim))
    if full <= kernel_config.threads
        @cuda(threads=full, blocks=(1, blocks_other...), shmem=2*full*sizeof(T), name="scan",
              partial_scan(f, output, input, Rdim, Rpre, Rpost, Rother, neutral, init, Val(true)))
    else
        # perform partial scans across the scanning dimension
        partial = prevpow(2, kernel_config.threads)
        blocks_dim = cld(length(Rdim), partial)
        @cuda(threads=partial, blocks=(blocks_dim, blocks_other...), shmem=2*partial*sizeof(T),
              partial_scan(f, output, input, Rdim, Rpre, Rpost, Rother, neutral, init, Val(true)))

        # get the total of each thread block (except the first) of the partial scans
        aggregates = fill(neutral, Base.setindex(size(input), blocks_dim, dims))
        copyto!(aggregates, selectdim(output, dims, partial:partial:length(Rdim)))

        # scan these totals to get totals for the entire partial scan
        accumulate!(f, aggregates, aggregates; dims=dims)

        # add those totals to the partial scan result
        # NOTE: we assume that this kernel requires fewer resources than the scan kernel.
        #       if that does not hold, launch with fewer threads and calculate
        #       the aggregate block index within the kernel itself.
        @cuda(threads=partial, blocks=(blocks_dim, blocks_other...),
              aggregate_partial_scan(f, output, aggregates, Rdim, Rpre, Rpost, Rother, init))

        CuArrays.unsafe_free!(aggregates)
    end

    return output
end


## Base interface

Base._accumulate!(op, output::CuArray, input::CuVector, dims::Nothing, init::Nothing) =
    scan!(op, output, input; dims=1)

Base._accumulate!(op, output::CuArray, input::CuArray, dims::Integer, init::Nothing) =
    scan!(op, output, input; dims=dims)

Base._accumulate!(op, output::CuArray, input::CuVector, dims::Nothing, init::Some) =
    scan!(op, output, input; dims=1, init=init)

Base._accumulate!(op, output::CuArray, input::CuArray, dims::Integer, init::Some) =
    scan!(op, output, input; dims=dims, init=init)

Base.accumulate_pairwise!(op, result::CuVector, v::CuVector) = accumulate!(op, result, v)
