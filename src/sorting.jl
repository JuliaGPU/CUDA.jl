"""
Quicksort!
Alex Ellison
@xaellison
Usage:
sort!(my_cuarray)
The main quicksort kernel uses dynamic parallelism. Let's call blocksize M. The
first part of the kernel bubble sorts M elements with maximal stride between
lo and hi. If the sublist is <= M elements, stride = 1 and no recursion
happens. Otherwise, we pick element lo + M ÷ 2 * stride as a pivot. This is
an efficient choice for random lists and pre-sorted lists.

Partition is done in stages:
1. Merge-sort batches of M values using their comparison to pivot as a key. The
   comparison alternates between < and <= with recursion depth. This makes no
   difference when there are many unique values, but when there are many
   duplicates, this effectively partitions into <, =, and >.
2. Consolidate batches. This runs inside the quicksort kernel.

Sublists (ranges of the list being sorted) are denoted by `lo` and one of
    `L` and `hi`. `lo` is an exclusive lower bound, `hi` is an
    inclusive upperboard, `L` is their difference.
`b_sums` is "batch sums", the number of values in a batch which are >= pivot or
    > pivot depending on the relevant `parity`
"""
module Sorting
using ..CUDA

#-------------------------------------------------------------------------------
# Integer arithmetic

"""
Returns smallest power of 2 < x
"""
function pow2_floor(x)
    out = 1
    while out * 2 <= x
        out *= 2
    end
    out
end

"""
For a batch of size `n` what is the lowest index of the batch `i` is in
"""
function batch_floor(idx, n)
    return idx - (idx - 1) % n
end

"""
For a batch of size `n` what is the highest index of the batch `i` is in
"""
function batch_ceil(idx, n)
    return idx + n - 1 - (idx - 1) % n
end

"""
GPU friendly step function (step at `i` = 1)
"""
function Θ(i)
    return 1 & (1 <= i)
end

"""
Suppose we are merging two lists of size n, each of which has all falses before
all trues. Together, they will be indexed 1:2n. This is a fast stepwise function
for the destination index of a value at index `x` in the concatenated input,
where `a` is the number of falses in the first half, b = n - a, and false is the
number of falses in the second half.
"""
function step_swap(x, a, b, c)
    return x + Θ(x - a) * b - Θ(x - (a + c)) * (b + c) + Θ(x - (a + b + c)) * c
end

"""
Generalizes `step_swap` for when the floor index is not 1
"""
function batch_step_swap(x, n, a, b, c)
    idx = (x - 1) % n + 1
    return batch_floor(x, n) - 1 + step_swap(idx, a, b, c)
end

@inline function flex_lt(a, b, eq)# where T
    eq ? a <= b : a < b
end

#-------------------------------------------------------------------------------
# Batch partitioning
"""
For thread `idx` with current value `value`, merge two batches of size `n` and
return the new value this thread takes. `sums` and `swap` are shared mem
"""
function merge_swap_shmem(value, idx, n, sums, swap)
    @inbounds begin
    sync_threads()
    b = sums[batch_floor(idx, 2 * n)]
    a = n - b
    d = sums[batch_ceil(idx, 2 * n)]
    c = n - d
    swap[idx] = value
    sync_threads()
    sums[idx] = d + b
    return swap[batch_step_swap(idx, 2 * n, a, b, c)]
    end
end

"""
Partition the region of `values` after index `lo` up to (inclusive) `hi` with
respect to `pivot`. This is done by a 'binary' merge sort, where each the values
are sorted by a boolean key: how they compare to `pivot`. The comparison is
affected by `parity`. See `flex_lt`. `swap` is an array for exchanging values
and `sums` is an array of Int32s used during the merge sort.

Uses block index to decide which values to operate on.
"""
@inline function batch_partition(values, pivot, swap, sums, lo, hi, parity)
    sync_threads()
    idx0 = Int(lo) + (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx0 <= hi
         swap[threadIdx().x] = values[idx0]
         sums[threadIdx().x] = 1 & flex_lt(pivot, swap[threadIdx().x], parity)
    else
        @inbounds sums[threadIdx().x] = 1
    end
    sync_threads()
    val = merge_swap_shmem(swap[threadIdx().x], threadIdx().x, 1, sums, swap)
    temp = 2
    while temp < blockDim().x
        val = merge_swap_shmem(val, threadIdx().x, temp, sums, swap)
        temp *= 2
    end
    sync_threads()

    if idx0 <= hi
         values[idx0] = val
    end
    sync_threads()
end

"""
Each block evaluates `batch_partition` on consecutive regions of length blockDim().x
from `lo` to `hi` of `values`.
"""
function partition_batches_kernel(values :: AbstractArray{T}, pivot, lo, hi, parity) where T
    swap = @cuDynamicSharedMem(T, blockDim().x, 4 * blockDim().x)
    sums = @cuDynamicSharedMem(Int32, blockDim().x)
    batch_partition(values, pivot, swap, sums, lo, hi, parity)
    return nothing
end

#-------------------------------------------------------------------------------
# Batch consolidation
"""
Finds the index in `array` of the last value <= `pivot` if `parity` = true or the
last value < `pivot` if `parity` = false.
Searches after index `lo` up to (inclusive) index `hi`
"""
function find_partition(array, pivot, lo, hi, parity) :: Int32
    low = lo + 1
    high = hi
    @inbounds while low <= high
        mid = (low + high) ÷ 2
        if flex_lt(pivot, array[mid], parity)
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
@inline function consolidate_batch_partition(vals :: AbstractArray{T}, pivot, lo, L, b_sums, parity) where T
    sync_threads()
    @inline N_b() = ceil(Int, L / blockDim().x)
    @inline batch(k) :: Int32 = threadIdx().x + k * blockDim().x

    my_iter = 0
    a = 0
    b = 0

    @inbounds for batch_i in 1:N_b()
        if batch_i % blockDim().x == 1
            if batch(my_iter) <= N_b()
                seek_lo = lo + (batch(my_iter) - 1) * blockDim().x
                seek_hi = lo + min(L, batch(my_iter) * blockDim().x)
                b_sums[threadIdx().x] = seek_hi - find_partition(vals, pivot, seek_lo, seek_hi, parity)
            end
            my_iter += 1
        end

        n_eff() = (batch_i != ceil(Int, L / blockDim().x) || L % blockDim().x == 0) ? blockDim().x : L % blockDim().x
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

#-------------------------------------------------------------------------------
# Sorting
"""
Performs bubble sort on `vals` starting after `lo` and going for min(`L`, `blockDim().x`)
elements spaced by `stride`. Good for sampling pivot values as well as short
sorts.
"""
@inline function bubble_sort(vals, swap, lo, L, stride)
    sync_threads()
    L = min(blockDim().x, L)
    @inbounds begin
    if threadIdx().x <= L
        swap[threadIdx().x] = vals[lo + threadIdx().x * stride]
    end
    sync_threads()
    for level in 0:L
        # get left/right neighbor depending on even/odd level
        buddy = threadIdx().x - 1 + 2 * (1 & (threadIdx().x % 2 != level % 2))
        if 1 <= buddy <= L && threadIdx().x <= L
            buddy_val = swap[buddy]
        end
        sync_threads()
        if 1 <= buddy <= L && threadIdx().x <= L
            if (threadIdx().x < buddy) != flex_lt(swap[threadIdx().x], buddy_val, false)
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
@inline function call_batch_partition(vals :: AbstractArray{T}, pivot, swap, b_sums, lo, hi, parity, sync :: Val{true}) where T
    L = hi - lo
    if threadIdx().x == 1
        @cuda blocks=ceil(Int, L / blockDim().x) threads=blockDim().x dynamic=true shmem=blockDim().x*(4+sizeof(T)) partition_batches_kernel(vals, pivot, lo, hi, parity)
        CUDA.device_synchronize()
    end
end

"""
Partition batches in a loop using a single block
"""
@inline function call_batch_partition(vals :: AbstractArray{T}, pivot, swap, b_sums, lo, hi, parity, sync :: Val{false}) where T
    for temp in lo:blockDim().x:hi
        batch_partition(vals, pivot, swap, b_sums, temp, min(hi, temp + blockDim().x), parity)
    end
end

"""
Perform quicksort on `vals` for the region with `lo` as an exclusive floor and
`hi` as an inclusive ceiling.
`parity` is a Val{Bool} which says whether to partition by < or <= with respect
to the pivot.
`sync_depth` is how many (more) levels of recursion with `qsort_kernel`
can be done before reaching `cudaLimitDevRuntimeSyncDepth`. From the host, this value
must not exceed that limit.

`sync` and enclosed type `S` determine how partition occurs:
 If `sync` is `true`:
This kernel partitions batches in a child kernel, synchronizes, and then
consolidates the batches. The benefit of this kernel is that it distributes
the work of partitioning batches across multiple SMs.
If `sync` is `false`:
This kernel partitions without launching any children kernels, then has recursive
`qsort_async_kernel` children for left and right partitions. `device_synchronize`
is never called from this kernel, so there is no practical limit on recursion.

To detect the scenario of all values in the region being the same, we have two
args: `prev_pivot` and `stuck`. If two consecutive partitions have the same pivot
and both failed to split the region in two, that means all the values are equal.
`stuck` is incremented when the pivot hasn't changed and partition = `lo` or `hi`.
If `stuck` reaches 2, recursion ends. `stuck` is initialized at -1 because
`prev_pivot` must be initialized to some value, and it's possible that the first
pivot will be that value, which could lead to an incorrectly early end to recursion
if we started `stuck` at 0.
"""
function qsort_kernel(vals :: AbstractArray{T}, lo, hi, parity , sync :: Val{S}, sync_depth, prev_pivot, stuck=-1) where {T, S}
    b_sums = @cuDynamicSharedMem(Int32, blockDim().x, 0)
    swap = @cuDynamicSharedMem(T, blockDim().x, 4 * blockDim().x)
    L = hi - lo

    #= step 1 bubble sort. It'll either finish sorting a subproblem or help
    select a pivot value =#
    bubble_sort(vals, swap, lo, L, L <= blockDim().x ? 1 : L ÷ blockDim().x)

    if L <= blockDim().x
        return
    end

    pivot = vals[lo + (blockDim().x ÷ 2) * (L ÷ blockDim().x)]

    # step 2: use pivot to partition into batches
    call_batch_partition(vals, pivot, swap, b_sums, lo, hi, parity, sync)

    #= step 3: consolidate the partitioned batches so that the sublist from
    [lo, hi) is partitioned, and the partition is stored in `partition`.
    Dispatching on P cleaner and faster than an if statement=#

    partition = consolidate_batch_partition(vals, pivot, lo, L, b_sums, parity)

    #= step 4: recursion =#
    if threadIdx().x == 1

        stuck = (pivot == prev_pivot && partition == lo || partition == hi) ? stuck + 1 : 0

        if stuck < 2 && partition > lo
            s = CuDeviceStream()
            if sync_depth > 1
                @cuda threads=blockDim().x dynamic=true stream=s shmem=blockDim().x*(4+sizeof(T)) qsort_kernel(vals, lo, partition, !parity, Val(true), sync_depth - 1, pivot, stuck)
            else
                @cuda threads=blockDim().x dynamic=true stream=s shmem=blockDim().x*(4+sizeof(T)) qsort_kernel(vals, lo, partition, !parity, Val(false), sync_depth - 1, pivot, stuck)
            end
            CUDA.unsafe_destroy!(s)
        end

        if stuck < 2 && partition < hi
            s = CuDeviceStream()
            if sync_depth > 1
                @cuda threads=blockDim().x dynamic=true stream=s shmem=blockDim().x*(4+sizeof(T)) qsort_kernel(vals, partition, hi, !parity, Val(true), sync_depth - 1, pivot, stuck)
            else
                @cuda threads=blockDim().x dynamic=true stream=s shmem=blockDim().x*(4+sizeof(T)) qsort_kernel(vals, partition, hi, !parity, Val(false), sync_depth - 1, pivot, stuck)
            end
            CUDA.unsafe_destroy!(s)
        end
    end

    return nothing
end

function quicksort!(c :: AbstractArray{T}) where T
    MAX_DEPTH = CUDA.limit(CUDA.LIMIT_DEV_RUNTIME_SYNC_DEPTH)
    N = length(c)

    kernel = @cuda launch=false qsort_kernel(c, 0, N, true, Val(MAX_DEPTH > 1), MAX_DEPTH, nothing)

    get_shmem(threads) = threads * (sizeof(Int32) + max(4, sizeof(T)))
    fun = kernel.fun
    config = launch_configuration(fun, shmem=threads->get_shmem(threads))
    threads = pow2_floor(config.threads)
    @assert threads <= config.threads

    kernel(c, 0, N, true, Val(MAX_DEPTH > 1), MAX_DEPTH, nothing;
           blocks=1, threads=threads, shmem=get_shmem(threads))
    synchronize()

    return c
end

function Base.sort!(c :: CuArray{T}; dims :: Integer, rev::Bool=false) where T
    # TODO: is it best to create a stream per call to `quicksort!` and sync after?
    nd = ndims(c)
    k = dims
    sz = size(c)

    1 <= k <= nd || throw(ArgumentError("dimension out of range"))

    remdims = ntuple(i -> i == k ? 1 : size(c, i), nd)
    for idx in CartesianIndices(remdims)
        if rev
            v = view(c, ntuple(i -> i == k ? range(sz[i], 1, step=-1) : idx[i], nd)...)
        else
            v = view(c, ntuple(i -> i == k ? Colon() : idx[i], nd)...)
        end
        quicksort!(v)
    end
    c
end

function Base.sort!(c :: CuVector{T}; rev=false) where T
    if rev
        quicksort!(view(c, range(length(c), 1, step=-1)))
    else
        quicksort!(c)
    end
    c
end

function Base.sort(c :: CuArray; by=identity, kwargs...)
    if by == identity
        return sort!(copy(c); kwargs...)
    else
        return map(x -> x[2], sort!(map(x -> (by(x), x), c); kwargs...))
    end
end

end
