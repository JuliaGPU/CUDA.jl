# sorting functionality

"""
The main quicksort kernel uses dynamic parallelism. Let's call blocksize `M`. The first part
of the kernel bubble sorts `M` elements with maximal stride between `lo` and `hi`. If the
sublist is <= `M` elements, `stride` = 1 and no recursion happens. Otherwise, we pick
element `lo + M ÷ 2 * stride` as a pivot. This is an efficient choice for random lists and
pre-sorted lists.

Partition is done in stages:
1. For batches of M values, cumsum how many > pivot are left of each index. The comparison
   alternates between < and <= with recursion depth. This makes no difference when there are
   many unique values, but when there are many duplicates, this effectively partitions into
   <, =, and >.
2. Consolidate batches. This runs inside the quicksort kernel.

Sublists (ranges of the list being sorted) are denoted by `lo` and one of `L` and `hi`. `lo`
is an exclusive lower bound, `hi` is an inclusive upperboard, `L` is their difference.
`b_sums` is "batch sums", the number of values in a batch which are >= pivot or > pivot
depending on the relevant `parity`

Originally developed by @xaellison (Alex Ellison).
"""
module QuickSortImpl

export quicksort!

using ..CUDA
using ..CUDA: i32


# Comparison

@inline function flex_lt(a, b, eq, lt, by)
    a′ = by(a)
    b′ = by(b)
    (eq && a′ == b′) || lt(a′, b′)
end


# Batch partitioning
"""
Performs in-place cumsum using shared memory. Intended for use with indexes
"""
function cumsum!(sums)
    shift = 1

    while shift < length(sums)
        to_add = 0
        @inbounds if threadIdx().x - shift > 0
            to_add = sums[threadIdx().x - shift]
        end

        sync_threads()
        @inbounds if threadIdx().x - shift > 0
            sums[threadIdx().x] += to_add
        end

        sync_threads()
        shift *= 2
    end
end

"""
Partition the region of `values` after index `lo` up to (inclusive) `hi` with
respect to `pivot`. Computes each value's comparison to pivot, performs a cumsum
of those comparisons, and performs one movement using shmem. Comparison is
affected by `parity`. See `flex_lt`. `swap` is an array for exchanging values
and `sums` is an array of Ints used during the merge sort.
Uses block y index to decide which values to operate on.
"""
@inline function batch_partition(values, pivot, swap, sums, lo, hi, parity,
                                 lt::F1, by::F2) where {F1,F2}
    sync_threads()
    blockIdx_yz = (blockIdx().z - 1i32) * gridDim().y + blockIdx().y
    idx0 = lo + (blockIdx_yz - 1i32) * blockDim().x + threadIdx().x
    @inbounds if idx0 <= hi
        val = values[idx0]
        comparison = flex_lt(pivot, val, parity, lt, by)
    end

    @inbounds if idx0 <= hi
         sums[threadIdx().x] = 1 & comparison
    else
         sums[threadIdx().x] = 1
    end
    sync_threads()

    cumsum!(sums)

    @inbounds if idx0 <= hi
        dest_idx = @inbounds if comparison
            blockDim().x - sums[end] + sums[threadIdx().x]
        else
            threadIdx().x - sums[threadIdx().x]
        end
        if dest_idx <= length(swap)
            swap[dest_idx] = val
        end
    end
    sync_threads()

    @inbounds if idx0 <= hi
         values[idx0] = swap[threadIdx().x]
    end
    sync_threads()
end

"""
Each block evaluates `batch_partition` on consecutive regions of length blockDim().x
from `lo` to `hi` of `values`.
"""
function partition_batches_kernel(values::AbstractArray{T}, pivot, lo, hi, parity, lt::F1,
                                  by::F2) where {T,F1,F2}
    sums = CuDynamicSharedArray(Int, blockDim().x)
    swap = CuDynamicSharedArray(T, blockDim().x, sizeof(sums))
    batch_partition(values, pivot, swap, sums, lo, hi, parity, lt, by)
    return
end


# Batch consolidation

"""
Finds the index in `array` of the last value <= `pivot` if `parity` = true or the
last value < `pivot` if `parity` = false.
Searches after index `lo` up to (inclusive) index `hi`
"""
function find_partition(array, pivot, lo, hi, parity, lt::F1, by::F2) where {F1,F2}
    low = lo + 1
    high = hi
    @inbounds while low <= high
        mid = (low + high) ÷ 2
        if flex_lt(pivot, array[mid], parity, lt, by)
            high = mid - 1
        else
            low = mid + 1
        end
    end
    return low - 1
end

"""
This assumes the region of `vals` of length `L` starting after `lo`
has been batch partitioned with respect to `pivot`. Further, it assumes that
these batches are of size `blockDim().x`.

Using 1 step per batch, consolidate these partitioned batches such that the
region is fully partitioned. Each step moves at most `blockDim().x` values.

`b_sums`: either shared memory or a global array which serves as scratch space
for storing the partition of each batch.

`parity`: see top docstring

Must only run on 1 SM.
"""
@inline function consolidate_batch_partition(vals::AbstractArray{T}, pivot, lo, L, b_sums,
                                             parity, lt::F1, by::F2) where {T,F1,F2}
    sync_threads()
    @inline N_b() = cld(L , blockDim().x)
    @inline batch(k) = threadIdx().x + k * blockDim().x

    my_iter = 0
    a = 0
    b = 0

    @inbounds for batch_i in 1:N_b()
        if batch_i % blockDim().x == 1
            if batch(my_iter) <= N_b()
                seek_lo = lo + (batch(my_iter) - 1) * blockDim().x
                seek_hi = lo + min(L, batch(my_iter) * blockDim().x)
                b_sums[threadIdx().x] =
                    seek_hi - find_partition(vals, pivot, seek_lo, seek_hi, parity, lt, by)
            end
            my_iter += 1
        end

        function n_eff()
            if batch_i != N_b() || L % blockDim().x == 0
                blockDim().x
            else
                L % blockDim().x
            end
        end

        sync_threads()
        d = b_sums[batch_i - (my_iter - 1) * blockDim().x]
        c = n_eff() - d
        to_move = min(b, c)
        sync_threads()
        if threadIdx().x <= to_move
            swap = vals[lo + a + threadIdx().x]
        end
        sync_threads()
        if threadIdx().x <= to_move
            vals[lo + a + threadIdx().x] = vals[lo + a + b + c - to_move + threadIdx().x]
        end
        sync_threads()
        if threadIdx().x <= to_move
            vals[lo + a + b + c - to_move + threadIdx().x] = swap
        end
        sync_threads()
        a += c
        b += d
    end

    sync_threads()
    return lo + a
end


# Sorting
"""
Finds the median of `vals` starting after `lo` and going for `blockDim().x`
elements spaced by `stride`. Performs bitonic sort in shmem, returns middle value.
Faster than bubble sort, but not as flexible. Does not modify `vals`
"""
function bitonic_median(vals::AbstractArray{T}, swap, lo, L, stride, lt::F1, by::F2) where {T,F1,F2}
    sync_threads()
    bitonic_lt(i1, i2) = @inbounds flex_lt(swap[i1 + 1], swap[i2 + 1], false, lt, by)

    @inbounds swap[threadIdx().x] = vals[lo + threadIdx().x * stride]
    sync_threads()

    log_blockDim = begin
        out = 0
        k = blockDim().x
        while k > 1
            k = k >> 1
            out += 1
        end
        out
    end

    log_k = 1
    while log_k <= log_blockDim
        k = 1 << log_k
        j = k ÷ 2

        while j > 0
            i = threadIdx().x - 1i32
            l = xor(i, j)
            to_swap = (i & k) == 0 && bitonic_lt(l, i) || (i & k) != 0 && bitonic_lt(i, l)
            to_swap = to_swap == (i < l)

            if to_swap
                @inbounds old_val = swap[l + 1]
            end
            sync_threads()
            if to_swap
                @inbounds swap[i + 1] = old_val
            end
            sync_threads()
            j = j ÷ 2
        end
        log_k += 1
    end
    sync_threads()
    return @inbounds swap[blockDim().x ÷ 2]
end

"""
Performs bubble sort on `vals` starting after `lo` and going for min(`L`, `blockDim().x`)
elements spaced by `stride`. Good for sampling pivot values as well as short sorts.
"""
@inline function bubble_sort(vals, swap, lo, L, stride, lt::F1, by::F2) where {F1,F2}
    sync_threads()
    L = min(blockDim().x, L)
    @inbounds begin
        if threadIdx().x <= L
            swap[threadIdx().x] = vals[lo + threadIdx().x * stride]
        end
        sync_threads()
        for level in 0:L
            # get left/right neighbor depending on even/odd level
            buddy = threadIdx().x - 1i32 + 2i32 * (1i32 & (threadIdx().x % 2i32 != level % 2i32))
            if 1 <= buddy <= L && threadIdx().x <= L
                 buddy_val = swap[buddy]
            end
            sync_threads()
            if 1 <= buddy <= L && threadIdx().x <= L
                is_left = threadIdx().x < buddy
                # flex_lt needs to handle equivalence in opposite ways for the
                # two threads in each swap pair. Otherwise, if there are two
                # different values with the same by, one will overwrite the other
                if is_left != flex_lt(swap[threadIdx().x], buddy_val, is_left, lt, by)
                    swap[threadIdx().x] = buddy_val
                end
            end
            sync_threads()
        end
        if threadIdx().x <= L
            vals[lo + threadIdx().x * stride] = swap[threadIdx().x]
        end
    end
    sync_threads()
end

"""
Launch batch partition kernel and sync
"""
@inline function call_batch_partition(vals::AbstractArray{T}, pivot, swap, b_sums, lo, hi,
                                      parity, sync::Val{true}, lt::F1, by::F2) where {T, F1, F2}
    L = hi - lo
    if threadIdx().x == 1
        blocks_y = cld(L, blockDim().x)

        # TODO: add wrappers
        device = Ref{Cint}()
        CUDA.cudaGetDevice(device)
        max_blocks_y = Ref{Cint}()
        CUDA.cudaDeviceGetAttribute(max_blocks_y, CUDA.cudaDevAttrMaxGridDimY, device[])

        blocks_z, blocks_y = fldmod1(blocks_y, max_blocks_y[])

        @cuda(blocks=(1,blocks_y,blocks_z), threads=blockDim().x, dynamic=true,
              shmem=blockDim().x*(sizeof(Int)+sizeof(T)),
              partition_batches_kernel(vals, pivot, lo, hi, parity, lt, by))
        device_synchronize()
    end

    # XXX: this global fence shouldn't be needed, according to the CUDA docs a sync_threads
    #      should be sufficient to propagate the global memory writes from the child grid
    #      to the (single-block) parent. See JuliaGPU/CUDA.jl#955 for more details.
    threadfence()
end

"""
Partition batches in a loop using a single block
"""
@inline function call_batch_partition(vals::AbstractArray{T}, pivot, swap, b_sums, lo, hi,
                                      parity, sync::Val{false}, lt::F1, by::F2) where {T, F1, F2}
    while lo <= hi
        batch_partition(vals, pivot, swap, b_sums, lo, min(hi, lo + blockDim().x), parity, lt, by)
        lo += blockDim().x
    end
end

"""
Quicksort recursion condition
For a full sort, `partial` is nothing so it shouldn't affect whether recursion
happens.
"""
function partial_range_overlap(lo, hi, partial::Nothing)
    true
end

"""
Quicksort recursion condition
If the domain to sort `lo` to `hi` overlaps with `partial`, then we should
do recursion on it, and this returns true (if not, then false)
"""
function partial_range_overlap(lo, hi, partial_k)
    return !(lo > last(partial_k) || hi < first(partial_k))
end

"""
Perform quicksort on dimension `dims` of `vals` for the region with `lo` as an exclusive
floor and `hi` as an inclusive ceiling. `parity` is a boolean which says whether to
partition by < or <= with respect to the pivot. `sync_depth` is how many (more) levels of
recursion with `qsort_kernel` can be done before reaching `cudaLimitDevRuntimeSyncDepth`.
From the host, this value must not exceed that limit.

`sync` and enclosed type `S` determine how partition occurs: If `sync` is `true`, the kernel
partitions batches in a child kernel, synchronizes, and then consolidates the batches. The
benefit of this kernel is that it distributes the work of partitioning batches across
multiple SMs. If `sync` is `false`, the kernel partitions without launching any child
kernels, then has recursive `qsort_kernel` children for left and right partitions.
`device_synchronize` is never called from this kernel, so there is no practical limit on
recursion.

To detect the scenario of all values in the region being the same, we have two args:
`prev_pivot` and `stuck`. If two consecutive partitions have the same pivot and both failed
to split the region in two, that means all the values are equal. `stuck` is incremented when
the pivot hasn't changed and partition = `lo` or `hi`. If `stuck` reaches 2, recursion ends.
`stuck` is initialized at -1 because `prev_pivot` must be initialized to some value, and
it's possible that the first pivot will be that value, which could lead to an incorrectly
early end to recursion if we started `stuck` at 0.
"""
function qsort_kernel(vals::AbstractArray{T,N}, lo, hi, parity, sync::Val{S}, sync_depth,
                      prev_pivot, lt::F1, by::F2, ::Val{dims}, partial=nothing,
                      stuck=-1) where {T, N, S, F1, F2, dims}
    b_sums = CuDynamicSharedArray(Int, blockDim().x)
    swap = CuDynamicSharedArray(T, blockDim().x, sizeof(b_sums))
    shmem = sizeof(b_sums) + sizeof(swap)
    L = hi - lo

    # extract the dimension that we need to sort (selected from the rest by the block x index)
    slice = if N == 1
        vals
    else
        # dimensions that are not part of the sort; index them using the block index
        otherdims = ntuple(i -> i == dims ? 1 : size(vals, i), N)
        other = CartesianIndices(otherdims)[blockIdx().x]

        # create a view that keeps the sorting dimension but indexes across the others
        slicedims = map(Base.Slice, axes(vals))
        idxs = ntuple(i->i==dims ? slicedims[i] : other[i], N)
        view(vals, idxs...)
    end

    # step 1: single block sort. It'll either finish sorting a subproblem or
    # help select a pivot value

    if L <= blockDim().x
        bubble_sort(slice, swap, lo, L, 1, lt, by)
        return
    end

    pivot = bitonic_median(slice, swap, lo, L, L ÷ blockDim().x, lt, by)

    # step 2: use pivot to partition into batches
    call_batch_partition(slice, pivot, swap, b_sums, lo, hi, parity, sync, lt, by)

    # step 3: consolidate the partitioned batches so that the sublist from [lo, hi) is
    #         partitioned, and the partition is stored in `partition`. Dispatching on P
    #         cleaner and faster than an if statement

    partition = consolidate_batch_partition(slice, pivot, lo, L, b_sums, parity, lt, by)

    # step 4: recursion
    if threadIdx().x == 1
        stuck = (pivot == prev_pivot && partition == lo || partition == hi) ? stuck + 1 : 0

        if stuck < 2 && partition > lo && partial_range_overlap(lo, partition, partial)
            s = CuDeviceStream()
            if S && sync_depth > 1
                @cuda(threads=blockDim().x, dynamic=true, stream=s, shmem=shmem,
                      qsort_kernel(slice, lo, partition, !parity, Val(true), sync_depth - 1,
                      pivot, lt, by, Val(1), partial, stuck))
            else
                @cuda(threads=blockDim().x, dynamic=true, stream=s, shmem=shmem,
                      qsort_kernel(slice, lo, partition, !parity, Val(false), sync_depth - 1,
                      pivot, lt, by, Val(1), partial, stuck))
            end
            CUDA.unsafe_destroy!(s)
        end

        if stuck < 2 && partition < hi && partial_range_overlap(partition, hi, partial)
            s = CuDeviceStream()
            if S && sync_depth > 1
                @cuda(threads=blockDim().x, dynamic=true, stream=s, shmem=shmem,
                      qsort_kernel(slice, partition, hi, !parity, Val(true), sync_depth - 1,
                      pivot, lt, by, Val(1), partial, stuck))
            else
                @cuda(threads=blockDim().x, dynamic=true, stream=s, shmem=shmem,
                      qsort_kernel(slice, partition, hi, !parity, Val(false), sync_depth - 1,
                      pivot, lt, by, Val(1), partial, stuck))
            end
            CUDA.unsafe_destroy!(s)
        end
    end

    return
end

function sort_args(args, partial_k::Nothing)
    return args
end

function sort_args(args, partial_k)
    return (args..., partial_k)
end

#function sort

function quicksort!(c::AbstractArray{T,N}; lt::F1, by::F2, dims::Int, partial_k=nothing,
                    block_size_shift=0) where {T,N,F1,F2}
    # XXX: after JuliaLang/CUDA.jl#2035, which changed the kernel state struct contents,
    #      the max depth needed to be reduced by 1 to avoid an illegal memory crash...
    max_depth = CUDA.limit(CUDA.LIMIT_DEV_RUNTIME_SYNC_DEPTH) - 1
    len = size(c, dims)

    1 <= dims <= N || throw(ArgumentError("dimension out of range"))
    otherdims = ntuple(i -> i == dims ? 1 : size(c, i), N)

    my_sort_args = sort_args((c, 0, len, true, Val(N==1 && max_depth > 1),
             max_depth, nothing, lt, by, Val(dims)), partial_k)

    kernel = @cuda launch=false qsort_kernel(my_sort_args...)

    get_shmem(threads) = threads * (sizeof(Int) + sizeof(T))
    config = launch_configuration(kernel.fun, shmem=threads->get_shmem(threads))
    threads = prevpow(2, config.threads)
    threads = threads >> block_size_shift   # for testing purposes

    kernel(my_sort_args...; blocks=prod(otherdims), threads, shmem=get_shmem(threads))

    return c
end

end

"""
This is an iterative bitonic sort that mimics a recursive version to support
non-power2 lengths.

Credit for the recursive form of this algorithm goes to:
https://www.inf.hs-flensburg.de/lang/algorithmen/sortieren/bitonic/oddn.htm

CUDA.jl implementation originally by @xaellison

Overview: `comparator_kernel` implements a layer of sorting network comparators
generally. The sort could run just by looping over `comparator`, but
`comparator_small_kernel` copies values into shmem and loops over several
comparators that don't need to access any values outside the range held in
shared memory. It provides a moderate speedup.

Notation:
`k`, `j` denote the level of the sorting network (equivalently, recursion depth).
`vals` is the array of values of type `T` that is either being `sort`-ed or `sortperm`-ed.
`inds` is an array of indices of type `J` that gets permuted in `sortperm!` (standard 1-indexed)
`i1`, `i2` index either `vals` or `inds` depending on the operation.
`lo`, `n`, and `m` are integers of type `I` used to denote/calculate ranges as
    described in the recursive algorithm link above. Note these follow the 0-indexing
    convention from the above source.
"""
module BitonicSortImpl

export bitonic_sort!

using ..CUDA
using ..CUDA: i32


# General functions

@inline two(::Type{Int}) = 2
@inline two(::Type{Int32}) = 2i32

@inline function gp2lt(x::Int)::Int
    x -= 1
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    x |= x >> 32
    xor(x, x >> 1)
end

@inline function gp2lt(x::Int32)::Int32
    x -= 1i32
    x |= x >> 1i32
    x |= x >> 2i32
    x |= x >> 4i32
    x |= x >> 8i32
    x |= x >> 16i32
    xor(x, x >> 1i32)
end

@inline function bisect_range(index::I, lo::I, n::I) where {I}
    if n <= one(I)
        return -one(I), -one(I)
    end
    m = gp2lt(n)
    if index < lo + m
        n = m
    else
        lo = lo + m
        n = n - m
    end
    return lo, n
end

@inline function rev_lt(a::T, b::T, lt, rev::Val{R}) where {T,R}
    if R
        return lt(b, a)
    else
        return lt(a, b)
    end
end

@inline function rev_lt(a::Tuple{T,J}, b::Tuple{T,J}, lt, rev::Val{R}) where {T,J,R}
    if R
        if a[1] == b[1]
            return a[2] < b[2]
        else
            return lt(b[1], a[1])
        end
    else
        return lt(a, b)
    end
end

@inline function extraneous_block(vals::AbstractArray, dims):: Bool
    other_linear_index = ((gridDim().z  ÷ blockDim().z) * (blockIdx().y - 1)) + blockIdx().z
    return other_linear_index > length(vals) ÷ size(vals)[dims]
end

@inline function extraneous_block(vals, dims)::Bool
    return extraneous_block(vals[1], dims)
end

# methods are defined for Val{1} because using view has 2x speed penalty for 1D arrays
@inline function view_along_dims(vals::AbstractArray{T, 1}, dimsval::Val{1}) where T
    return vals
end

@inline function view_along_dims(vals::Tuple{AbstractArray{T,1},Any}, dimsval::Val{1}) where T
    return vals[1], view_along_dims(vals[2], dimsval)
end


@inline function view_along_dims(vals::AbstractArray{T, N}, ::Val{dims}) where {T,N,dims}
    otherdims = ntuple(i -> i == dims ? 1 : size(vals, i), N)
    other_linear_index = ((gridDim().z  ÷ blockDim().z) * (blockIdx().y - 1)) + blockIdx().z
    other = CartesianIndices(otherdims)[other_linear_index]
    # create a view that keeps the sorting dimension but indexes across the others
    slicedims = map(Base.Slice, axes(vals))
    idxs = ntuple(i->i==dims ? slicedims[i] : other[i], N)
    return view(vals, idxs...)
end

@inline function view_along_dims(vals, dimsval::Val{dims}) where dims
    return vals[1], view_along_dims(vals[2], dimsval)
end


# Functions specifically for "large" bitonic steps (those that cannot use shmem)

@inline function compare!(vals::AbstractArray{T, N}, i1::I, i2::I, dir::Bool, by, lt, rev) where {T,I,N}
    i1′, i2′ = i1 + one(I), i2 + one(I)
    @inbounds if dir != rev_lt(by(vals[i1′]), by(vals[i2′]), lt, rev)
        vals[i1′], vals[i2′] = vals[i2′], vals[i1′]
    end
end

@inline function compare!(vals_inds::Tuple, i1::I, i2::I, dir::Bool, by, lt,
                          rev) where {I}
    i1′, i2′ = i1 + one(I), i2 + one(I)
    vals, inds = vals_inds
    # comparing tuples of (value, index) guarantees stability of sort
    @inbounds if dir != rev_lt((by(vals[inds[i1′]]), inds[i1′]),
                               (by(vals[inds[i2′]]), inds[i2′]), lt, rev)
        inds[i1′], inds[i2′] = inds[i2′], inds[i1′]
    end
end


@inline function get_range_part1(n::I, index::I, k::I)::Tuple{I,I,Bool} where {I}
    lo = zero(I)
    dir = true
    for iter = one(I):k-one(I)
        if n <= one(I)
            return -one(I), -one(I), false
        end

        if index < lo + n ÷ two(I)
            n = n ÷ two(I)
            dir = !dir
        else
            lo = lo + n ÷ two(I)
            n = n - n ÷ two(I)
        end
    end
    return lo, n, dir
end

@inline function get_range_part2(lo::I, n::I, index::I, j::I)::Tuple{I,I} where {I}
    for iter = one(I):j-one(I)
        lo, n = bisect_range(index, lo, n)
    end
    return lo, n
end

"""
Determines parameters for swapping when the grid index directly maps to an
Array index for swapping
"""
@inline function get_range(n, index, k, j)
    lo, n, dir = get_range_part1(n, index, k)
    lo, n = get_range_part2(lo, n, index, j)
    return lo, n, dir
end

"""
Performs a step of bitonic sort requiring swaps between indices further apart
than the size of block allows (eg, 1 <--> 10000)

The grid index directly maps to the index of `c` that will be used in the swap.

Note that to avoid synchronization issues, only one thread from each pair of
indices being swapped will actually move data.
"""
function comparator_kernel(vals, length_vals::I, k::I, j::I, by::F1, lt::F2,
                           rev, dimsval::Val{dims}) where {I,F1,F2,dims}
    if extraneous_block(vals, dims)
        return nothing
    end

    index = (blockDim().x * (blockIdx().x - one(I))) + threadIdx().x - one(I)

    slice = view_along_dims(vals, dimsval)

    lo, n, dir = get_range(length_vals, index, k, j)

    if !(lo < zero(I) || n < zero(I)) && !(index >= length_vals)
        m = gp2lt(n)
        if lo <= index < lo + n - m
            i1, i2 = index, index + m
            @inbounds compare!(slice, i1, i2, dir, by, lt, rev)
        end
    end
    return
end


# Functions for "small" bitonic steps (those that can use shmem)

@inline function compare_small!(vals::AbstractArray{T}, i1::I, i2::I, dir::Bool, by, lt,
                                rev) where {T,I}
    i1′, i2′ = i1 + one(I), i2 + one(I)
    @inbounds if dir != rev_lt(by(vals[i1′]), by(vals[i2′]), lt, rev)
        vals[i1′], vals[i2′] = vals[i2′], vals[i1′]
    end
end

@inline function compare_small!(vals_inds::Tuple, i1::I, i2::I, dir::Bool, by, lt,
                                rev) where {I}
    i1′, i2′ = i1 + one(I), i2 + one(I)
    vals, inds = vals_inds
    # comparing tuples of (value, index) guarantees stability of sort
    @inbounds if dir != rev_lt((by(vals[i1′]), inds[i1′]),
                               (by(vals[i2′]), inds[i2′]), lt, rev)
        vals[i1′], vals[i2′] = vals[i2′], vals[i1′]
        inds[i1′], inds[i2′] = inds[i2′], inds[i1′]
    end
end

"""
For each thread in the block, "re-compute" the range which would have been
passed in recursively. This range only depends on the block, and guarantees
all threads perform swaps accessible using shmem.

Various negative exit values just for debugging.
"""
@inline function block_range(n::I, block_index::I, k::I, j::I)::Tuple{I,I,Bool} where {I}
    lo = zero(I)
    dir = true
    tmp = block_index * two(I)

    # analogous to `get_range_part1`
    for iter = one(I):(k-one(I))
        tmp ÷= two(I)
        if n <= one(I)
            return -one(I), -one(I), false
        end

        if tmp % two(I) == zero(I)
            n = n ÷ two(I)
            dir = !dir
        else
            lo = lo + n ÷ two(I)
            n = n - n ÷ two(I)
        end
    end

    # analogous to `get_range_part2`
    for iter = one(I):(j-one(I))
        tmp ÷= two(I)
        if n <= one(I)
            return -one(I), -one(I), false
        end

        m = gp2lt(n)
        if tmp % two(I) == zero(I)
            n = m
        else
            lo = lo + m
            n = n - m
        end

    end
    if zero(I) <= n <= one(I)
        return -one(I), -one(I), false
    end
    return lo, n, dir
end

"""
For sort/sort! `c`, allocate and return shared memory view of `c`
Each view is indexed along block x dim: one view per pseudo-block
`index` is expected to be from a 0-indexing context
"""
@inline function initialize_shmem!(vals::AbstractArray{T}, index::I, in_range,
                                   offset = zero(I)) where {T,I}
    swap = CuDynamicSharedArray(T, (blockDim().x, blockDim().y), offset)
    if in_range
        @inbounds swap[threadIdx().x, threadIdx().y] = vals[index+one(I)]
    end
    sync_threads()
    return @inbounds @view swap[:, threadIdx().y]
end

"""
For sortperm/sortperm!, allocate and return shared memory views of `c` and index
array. Each view is indexed along block x dim: one view per pseudo-block.
`index` is expected to be from a 0-indexing context, but the indices stored in
`val_inds` are expected to be 1-indexed
"""
@inline function initialize_shmem!(vals_inds::Tuple{AbstractArray{T},AbstractArray{J}},
                                   index, in_range) where {T,J}
    offset = prod(blockDim()) * sizeof(J)
    vals, inds = vals_inds
    inds_view = initialize_shmem!(inds, index, in_range)
    vals_view = initialize_shmem!(vals, inds_view[threadIdx().x] - one(J), in_range, offset)
    return vals_view, inds_view
end

"""
For sort/sort!, copy shmem view `swap` back into global array `c`
`index` is expected to be from a 0-indexing context
"""
@inline function finalize_shmem!(vals::AbstractArray, swap::AbstractArray, index::I,
                                 in_range::Bool) where {I}
    if in_range
        @inbounds vals[index+one(I)] = swap[threadIdx().x]
    end
end

"""
For sortperm/sortperm!, copy shmem view `swap` back to global index array
`index` is expected to be from a 0-indexing context, but the indices stored in
`val_inds` are expected to be 1-indexed
"""
@inline function finalize_shmem!(vals_inds::Tuple, swap::Tuple, index, in_range::Bool)
    vals, inds = vals_inds
    swap_vals, swap_inds = swap
    finalize_shmem!(inds, swap_inds, index, in_range)
end

"""
Performs consecutive steps of bitonic sort requiring swaps between indices no
further apart than the size of block allows. This effectively moves part of the
inner loop (over j, below) inside of a kernel to minimize launches and do
swaps in shared mem.

Note that the x dimension of a thread block is treated as a comparator,
so when the maximum size of a comparator in this kernel is small, multiple
may be executed along the block y dimension, allowing for higher occupancy.
These threads in a block with the same threadIdx().x are a 'pseudo-block',
and are indexed by `pseudo_block_idx`.

Unlike `comparator_kernel`, a thread's grid_index does not directly map to the
index of `c` it will read from. `block_range` gives gives each pseudo-block
a unique  range of indices corresponding to a comparator in the sorting network.

Note that this moves the array values copied within shmem, but doesn't copy them
back to global the way it does for indices.
"""
function comparator_small_kernel(vals, length_vals::I, k::I, j_0::I, j_f::I,
                                 by::F1, lt::F2, rev, dimsval::Val{dims}) where {I,F1,F2,dims}
    if extraneous_block(vals, dims)
        return nothing
    end
    slice = view_along_dims(vals, dimsval)
    pseudo_block_idx = (blockIdx().x - one(I)) * blockDim().y + threadIdx().y - one(I)
    # immutable info about the range used by this kernel
    _lo, _n, dir = block_range(length_vals, pseudo_block_idx, k, j_0)
    index = _lo + threadIdx().x - one(I)
    in_range = (threadIdx().x <= _n && _lo >= zero(I))

    swap = initialize_shmem!(slice, index, in_range)

    # mutable copies for pseudo-recursion
    lo, n = _lo, _n

    for j = j_0:j_f
        if !(lo < zero(I) || n < zero(I)) && in_range
            m = gp2lt(n)
            if lo <= index < lo + n - m
                i1, i2 = index - _lo, index - _lo + m
                compare_small!(swap, i1, i2, dir, by, lt, rev)
            end
        end
        lo, n = bisect_range(index, lo, n)
        sync_threads()
    end

    finalize_shmem!(slice, swap, index, in_range)
    return
end


# Host side code
function bitonic_shmem(c::AbstractArray{T}, threads) where {T}
    return prod(threads) * sizeof(T)
end

function bitonic_shmem(c, threads)
    return prod(threads) * sum(map(a -> sizeof(eltype(a)), c))
end

"""
Call bitonic sort on `c` which can be a CuArray of values to `sort!` or a tuple
of values and an index array for doing `sortperm!`. Cannot provide a stable
`sort!` although `sortperm!` is properly stable. To reverse, set `rev=true`
rather than `lt=!isless` (otherwise stability of sortperm breaks down).
"""
function bitonic_sort!(c; by = identity, lt = isless, rev = false, dims=1)
    c_len, otherdims_len = if typeof(c) <: Tuple
        size(c[1])[dims], length(c[1]) ÷ size(c[1])[dims]
    else
        size(c)[dims], length(c) ÷ size(c)[dims]
    end

    # compile kernels (using Int32 for indexing, if possible, yielding a 70% speedup)
    I = c_len <= typemax(Int32) ? Int32 : Int

    args1 = (c, I(c_len), one(I), one(I), one(I), by, lt, Val(rev), Val(dims))
    kernel1 = @cuda launch=false comparator_small_kernel(args1...)
    config1 = launch_configuration(kernel1.fun, shmem = threads -> bitonic_shmem(c, threads))
    # blocksize for kernel1 MUST be a power of 2
    threads1 = prevpow(2, config1.threads)

    args2 = (c, I(c_len), one(I), one(I), by, lt, Val(rev), Val(dims))
    kernel2 = @cuda launch=false comparator_kernel(args2...)
    config2 = launch_configuration(kernel2.fun, shmem = threads -> bitonic_shmem(c, threads))
    threads2 =  config2.threads

    # determines cutoff for when to use kernel1 vs kernel2
    log_threads = threads1 |> log2 |> Int

    # These two outer loops are the same as the serial version outlined here:
    # https://en.wikipedia.org/wiki/Bitonic_sorter#Example_code
    # Notation: our k/j are the base2 logs of k/j in Wikipedia example
    k0 = ceil(Int, log2(c_len))
    for k = k0:-1:1
        j_final = 1 + k0 - k

        # non-sorting dims are put into blocks along grid y/z. Using sqrt minimizes wasted blocks
        other_block_dims = Int(ceil(sqrt(otherdims_len))), Int(ceil(sqrt(otherdims_len)))

        for j = 1:j_final
            args1 = (c, I.((c_len, k, j, j_final))..., by, lt, Val(rev), Val(dims))
            args2 = (c, I.((c_len, k, j))..., by, lt, Val(rev), Val(dims))
            if k0 - k - j + 2 <= log_threads
                # pseudo_block_length = max(nextpow(2, length(comparator))
                # for all comparators in this layer of the network)
                pseudo_block_length = 1 << abs(j_final + 1 - j)
                # N_pseudo_blocks = how many pseudo-blocks are in this layer of the network
                N_pseudo_blocks = nextpow(2, c_len) ÷ pseudo_block_length
                pseudo_blocks_per_block = threads1 ÷ pseudo_block_length

                # grid dimensions
                N_blocks = max(1, N_pseudo_blocks ÷ pseudo_blocks_per_block), other_block_dims...
                block_size = pseudo_block_length, threads1 ÷ pseudo_block_length
                kernel1(args1...; blocks=N_blocks, threads=block_size,
                        shmem=bitonic_shmem(c, block_size))
                break
            else
                N_blocks = cld(c_len, threads2), other_block_dims...
                kernel2(args2...; blocks = N_blocks, threads=threads2)
            end
        end
    end

    return c
end
end


# Base interface implementation

using .BitonicSortImpl
using .QuickSortImpl


abstract type SortingAlgorithm end
struct QuickSortAlg <: SortingAlgorithm end
struct BitonicSortAlg <: SortingAlgorithm end

const QuickSort = QuickSortAlg()
const BitonicSort = BitonicSortAlg()


function Base.sort!(c::AnyCuVector, alg::QuickSortAlg; lt=isless, by=identity, rev=false)
    # for reverse sorting, invert the less-than function
    if rev
        lt = !lt
    end

    quicksort!(c; lt, by, dims=1)
    return c
end

function Base.sort!(c::AnyCuArray, alg::BitonicSortAlg; kwargs...)
    return bitonic_sort!(c; kwargs...)
end

function Base.sort!(c::AnyCuArray; alg::SortingAlgorithm = BitonicSort, kwargs...)
    return sort!(c, alg; kwargs...)
end

function Base.sort(c::AnyCuArray; kwargs...)
    return sort!(copy(c); kwargs...)
end

function Base.partialsort!(c::AnyCuVector, k::Union{Integer, OrdinalRange},
                           alg::BitonicSortAlg; lt=isless, by=identity, rev=false)

    sort!(c, alg; lt, by, rev)
    return @allowscalar copy(c[k])
end

function Base.partialsort!(c::AnyCuVector, k::Union{Integer, OrdinalRange},
                           alg::QuickSortAlg; lt=isless, by=identity, rev=false)
    # for reverse sorting, invert the less-than function
    if rev
        lt = !lt
    end

    function out(k::OrdinalRange)
        return copy(c[k])
    end

    # work around disallowed scalar index
    function out(k::Integer)
        return Array(c[k:k])[1]
    end

    quicksort!(c; lt, by, dims=1, partial_k=k)
    return out(k)
end

function Base.partialsort!(c::AnyCuArray, k::Union{Integer, OrdinalRange};
                           alg::SortingAlgorithm=BitonicSort, kwargs...)
    return partialsort!(c, k, alg; kwargs...)
end

function Base.partialsort(c::AnyCuArray, k::Union{Integer, OrdinalRange}; kwargs...)
    return partialsort!(copy(c), k; kwargs...)
end

function Base.sortperm!(ix::AnyCuArray, A::AnyCuArray; initialized=false, kwargs...)
    if axes(ix) != axes(A)
        throw(ArgumentError("index array must have the same size/axes as the source array, $(axes(ix)) != $(axes(A))"))
    end

    if !initialized
        ix .= LinearIndices(A)
    end
    bitonic_sort!((A, ix); kwargs...)
    return ix
end

function Base.sortperm(c::AnyCuVector; kwargs...)
    sortperm!(CuArray(1:length(c)), c; initialized=true, kwargs...)
end

function Base.sortperm(c::AnyCuArray; dims, kwargs...)
    # Base errors for Matrices without dims arg, we should too
    sortperm!(reshape(CuArray(1:length(c)), size(c)), c; initialized=true, dims, kwargs...)
end
