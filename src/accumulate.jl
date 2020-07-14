# scan and accumulate

## COV_EXCL_START

# Prefix scan using warp intrinsics
# https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
# 
#
# TODOs:
# - multiple elements per thread (performance)
# - custom launch config (performance)

# Scan entire warp using shfl intrinsics, unrolled for warpsize() = 32
@inline function scan_warp(op, val, lane)
    mask = typemax(UInt32)

    left = shfl_up_sync(mask, val, 1)
    lane > 1 && (val = op(left, val))

    left = shfl_up_sync(mask, val, 2)
    lane > 2 && (val = op(left, val))

    left = shfl_up_sync(mask, val, 4)
    lane > 4 && (val = op(left, val))

    left = shfl_up_sync(mask, val, 8)
    lane > 8 && (val = op(left, val))

    left = shfl_up_sync(mask, val, 16)
    lane > 16 && (val = op(left, val))
    return val
end

# Scan warp without shfl intrinsics for complicated datatypes

@inline function scan_warp(op, val, lane, thread, cache)
    @inbounds begin
    sync_warp()
    if lane > 1 
        val = op(cache[thread - 1], val)
        cache[thread] = val
    end
    sync_warp()
    if lane > 2 
        val = op(cache[thread - 2], val)
        cache[thread] = val
    end
    sync_warp()
    if lane > 4 
        val = op(cache[thread - 4], val)
        cache[thread] = val
    end
    sync_warp()
    if lane > 8 
        val = op(cache[thread - 8], val)
        cache[thread] = val
    end
    sync_warp()
    if lane > 16 
        val = op(cache[thread - 16], val)
        cache[thread] = val
    end
    sync_warp()
    return val
end
end


function partial_scan!(op::Function, output::AbstractArray{T}, input::AbstractArray,
                      aggregates::Union{Nothing, AbstractArray{T}}, Rdim, Rpre, 
                      Rpost, Rother, neutral::T, init, ::Val{inclusive}=Val(true), 
                      ::Val{shuffle}=Val(true)) where {T, inclusive, shuffle}
    threads = blockDim().x
    thread = threadIdx().x
    block = blockIdx().x
    wid, lane = fldmod1(thread, warpsize())

    # cache: storage for non shuffle kernels
    partial_sums = @cuDynamicSharedMem(T, 32)
    cache = @cuDynamicSharedMem(T, threads*!shuffle)

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

    # load input into shared memory (apply `op` to have the correct type)
    value = if i <= length(Rdim)
        op(neutral, input[Ipre, i, Ipost])
    else
        op(neutral, neutral)
    end

    init !== nothing && i == 1 && (value = op(init, value))

    if shuffle
        value = scan_warp(op, value, lane)
    else
        cache[thread] = value
        value = scan_warp(op, value, lane, thread, cache)
        sync_threads()
    end

    lane == warpsize() && (partial_sums[wid] = value)

    sync_threads()

    # 1st warp computes sum
    # works because 32*32 = 1024 = max threads in a block, 
    if wid == 1 && shuffle
        p_sum = partial_sums[lane]
        p_sum = scan_warp(op, p_sum, lane)
        partial_sums[lane] = p_sum
    elseif wid == 1 
        p_sum = partial_sums[lane]
        p_sum = scan_warp(op, p_sum, lane, thread, partial_sums)
        partial_sums[lane] = p_sum
    end

    sync_threads()

    wid > 1 && (value = op(partial_sums[wid - 1], value))
    
    # write results to device memory
    @inbounds if i <= length(Rdim)
        output[Ipre, i, Ipost] = value
    end

    if aggregates !== nothing && thread == threads
        aggregates[Ipre, blockIdx().x ,Ipost] = value
    end
end
    return 
end

# aggregate the result of a partial scan by applying preceding block aggregates
function aggregate_partial_scan!(op::Function, output::AbstractArray,
                                aggregates::AbstractArray, Rdim, Rpre, Rpost, Rother)
   
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
        output[Ipre, i, Ipost] = val
    end
    return
end

## COV_EXCL_STOP

function scan!(f::Function, output::AnyCuArray{T}, input::AnyCuArray{T2};
               dims::Integer = 1, init=nothing, neutral=GPUArrays.neutral_element(f, T)) where {T, T2}
    dims > 0 || throw(ArgumentError("dims must be a positive integer"))
    inds_t = axes(input)
    axes(output) == inds_t || throw(DimensionMismatch("shape of B must match A"))
    dims > ndims(input) && return copyto!(output, input)
    isempty(inds_t[dims]) && return output
    init !== nothing && (init = convert(T, init))

    shuffle = true
    shuffle &= capability(device()) >= v"3.0"
    shuffle &= T in (Bool, Int32, Int64, Float32, Float64, ComplexF32, ComplexF64)

    f = cufunc(f)

    # iteration domain across the main dimension
    Rdim = CartesianIndices((size(input, dims),))

    # iteration domain for the other dimensions
    Rpre = CartesianIndices(size(input)[1:dims-1])
    Rpost = CartesianIndices(size(input)[dims+1:end])
    Rother = CartesianIndices((length(Rpre), length(Rpost)))

    # determine how many threads we can launch for the scan kernel
    args = (f, output, input, output, Rdim, Rpre, Rpost, Rother, neutral, init, Val(true), Val(shuffle))
    kernel = @cuda(launch=false, partial_scan!(args...))
    shmem_calc = (threads)->((32 + threads*!shuffle)*sizeof(T))
    kernel_config = launch_configuration(kernel.fun; shmem=shmem_calc)

    # determine the grid layout to cover the other dimensions
    if length(Rother) > 1
        dev = device()
        max_other_blocks = attribute(dev, DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)
        blocks_other = (Base.min(length(Rother), max_other_blocks),
                        cld(length(Rother), max_other_blocks))
    else
        blocks_other = (1, 1)
    end

    threads = nextwarp(device(), Base.min(kernel_config.threads, length(Rdim)))
    blocks = (cld(length(Rdim), threads), blocks_other...)
    aggregates = if length(Rdim) > threads
            aggregate_dims = [size(output)...]
            aggregate_dims[dims] = blocks[1]
            similar(output, Tuple(aggregate_dims))
        else
            nothing
    end
    shmem = shmem_calc(threads)

    args = (f, output, input, aggregates, Rdim, Rpre, Rpost, Rother, neutral, init, Val(true), Val(shuffle))
    @cuda(threads=threads, blocks=blocks, shmem=shmem, partial_scan!(args...))

    if length(Rdim) > threads
        scan!(f, aggregates, aggregates, dims=dims)
        args = (f, output, aggregates, Rdim, Rpre, Rpost, Rother)
        @cuda(threads=threads, blocks=blocks, shmem=shmem, aggregate_partial_scan!(args...))
    end

    aggregates !== nothing && unsafe_free!(aggregates)
    return output
end


## Base interface

Base._accumulate!(op, output::AnyCuArray, input::AnyCuVector, dims::Nothing, init::Nothing) =
    scan!(op, output, input; dims=1)

Base._accumulate!(op, output::AnyCuArray, input::AnyCuArray, dims::Integer, init::Nothing) =
    scan!(op, output, input; dims=dims)

Base._accumulate!(op, output::AnyCuArray, input::CuVector, dims::Nothing, init::Some) =
    scan!(op, output, input; dims=1, init=init.value)

Base._accumulate!(op, output::AnyCuArray, input::AnyCuArray, dims::Integer, init::Some) =
    scan!(op, output, input; dims=dims, init=init.value)

Base.accumulate_pairwise!(op, result::AnyCuVector, v::AnyCuVector) = accumulate!(op, result, v)
