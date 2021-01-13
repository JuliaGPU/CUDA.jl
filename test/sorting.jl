"""
Unit tests for quicksort.
"""

using DataStructures, Random, Test, CUDA.Sorting
import CUDA.Sorting: pow2_floor, Θ, flex_lt, find_partition,
        partition_batches_kernel, consolidate_batch_partition

@testset "Quicksort Integer Functions" begin
@test pow2_floor(1) == 1
@test pow2_floor(2) == 2
@test pow2_floor(3) == 2
@test pow2_floor(5) == 4
@test pow2_floor(8) == 8

@test Θ(0) == 0
@test Θ(1) == 1
@test Θ(2) == 1

@test flex_lt(1, 2, false) == true
@test flex_lt(1, 2, true) == true
@test flex_lt(2, 2, false) == false
@test flex_lt(2, 2, true) == true
@test flex_lt(3, 2, false) == false
@test flex_lt(3, 2, true) == false

@test find_partition([1, 2, 2, 3, 4, 1, 2, 2, 3, 4], 3, 0, 5, false) == 4
@test find_partition([1, 2, 2, 3, 4, 1, 2, 2, 3, 4], 3, 5, 10, false) == 9
@test find_partition([1, 2, 2, 3, 4, 1, 2, 2, 3, 4], 3, 0, 5, true) == 3
@test find_partition([1, 2, 2, 3, 4, 1, 2, 2, 3, 4], 3, 5, 10, true) == 8
end

function test_batch_partition(T, N, lo, hi, seed)
    my_range = lo + 1 : hi
    Random.seed!(seed)
    original = rand(T, N)
    A = CuArray(original)

    pivot = rand(original[my_range])
    block_N, block_dim = -1, -1

    kernel = @cuda launch=false partition_batches_kernel(A, pivot, lo, hi, true)

    get_shmem(threads) = threads * (sizeof(Int32) + max(4, sizeof(T)))
    config = launch_configuration(kernel.fun, shmem=threads->get_shmem(threads), max_threads=1024)

    threads = pow2_floor(config.threads)
    blocks = ceil(Int, (hi - lo) ./ threads)
    block_N = blocks
    block_dim = threads
    @assert block_dim >= 32 "This test assumes block size can be >= 32"

    kernel(A, pivot, lo, hi, true;
           threads=threads, blocks=blocks, shmem=get_shmem(threads))
    synchronize()

    post_sort = Array(A)

    sort_match = true

    for block in 1:block_N
        block_range = lo + 1 + (block - 1) * block_dim: min(hi, lo + block * block_dim)
        temp = original[block_range]
        #= this shows that batch partitioning is a stable sort where key for
        each value v is whether v > or <= pivot =#
        expected_sort = vcat(filter(x -> x < pivot, temp), filter(x -> x >= pivot, temp))
        sort_match &= post_sort[block_range] == expected_sort
    end

    @test sort_match
end

@testset "Quicksort batch partition" begin
test_batch_partition(Int8, 10000, 2000, 6000, 0)
test_batch_partition(Int8, 10000, 2000, 6000, 1)
test_batch_partition(Int8, 10000000, 0, 10000000, 0)
test_batch_partition(Int8, 10000000, 5000, 500000, 0)
test_batch_partition(Int8, 10000, 0, 10000, 0)
test_batch_partition(Int8, 10000, 2000, 6000, 0)
test_batch_partition(Int8, 10000, 2000, 6000, 1)
test_batch_partition(Int8, 10000000, 0, 10000000, 0)
test_batch_partition(Int8, 10000000, 5000, 500000, 0)

test_batch_partition(Float32, 10000, 0, 10000, 0)
test_batch_partition(Float32, 10000, 2000, 6000, 0)
test_batch_partition(Float32, 10000, 2000, 6000, 1)
test_batch_partition(Float32, 10000000, 0, 10000000, 0)
test_batch_partition(Float32, 10000000, 5000, 500000, 0)
test_batch_partition(Float32, 10000, 0, 10000, 0)
test_batch_partition(Float32, 10000, 2000, 6000, 0)
test_batch_partition(Float32, 10000, 2000, 6000, 1)
test_batch_partition(Float32, 10000000, 0, 10000000, 0)
test_batch_partition(Float32, 10000000, 5000, 500000, 0)
end

function test_consolidate_kernel(vals, pivot, my_floor, L, b_sums, dest, parity)
    i = threadIdx().x
    p = consolidate_batch_partition(vals, pivot, my_floor, L, b_sums, parity)
    if i == 1
        dest[1] = p
    end
    return nothing
end

function test_consolidate_partition(T, N, lo, hi, seed, block_dim)
    # assuming partition_batches works, we can validate consolidate by
    # checking that together they partition a large domain
    my_range = lo + 1 : hi
    Random.seed!(seed)
    original = rand(T, N)
    A = CuArray(original)
    pivot = rand(original[my_range])

    threads = blocks = -1
    sums = CuArray(zeros(Int32, ceil(Int, hi - lo / block_dim)))

    kernel = @cuda launch=false partition_batches_kernel(A, pivot, lo, hi, true)

    get_shmem(threads) = threads * (sizeof(Int32) + max(4, sizeof(T)))
    config = launch_configuration(kernel.fun, shmem=threads->get_shmem(threads), max_threads=1024)

    threads = isnothing(block_dim) ? pow2_floor(config.threads) : block_dim
    blocks = ceil(Int, (hi - lo) ./ threads)

    kernel(A, pivot, lo, hi, true; threads=threads, blocks=blocks, shmem=get_shmem(threads))
    synchronize()
    dest = CuArray(zeros(Int32, 1))

    @cuda threads=threads test_consolidate_kernel(A, pivot, lo, hi - lo, sums, dest, true)
    synchronize()

    partition = Array(dest)[1]
    temp = original[my_range]
    post_sort = Array(A)
    #= consolidation is a highly unstable sort (again, by pivot comparison as
    the key) so we compare by counting each element =#
    cc(x) = x |> counter |> collect |> sort
    @test cc(original) == cc(post_sort)
    @test all(post_sort[lo + 1 : partition] |> cc .== filter(x -> x < pivot, temp) |> cc)
    @test all(post_sort[partition + 1 : hi] |> cc .== filter(x -> x >= pivot, temp) |> cc)
end

@testset "Quicksort consolidate partition" begin
test_consolidate_partition(Int8, 10000, 0, 10000, 0, 16)
test_consolidate_partition(Int8, 10000, 0, 10000, 0, 32)
test_consolidate_partition(Int8, 10000, 0, 10000, 0, 64)
test_consolidate_partition(Int8, 10000, 9, 6333, 0, 16)
test_consolidate_partition(Int8, 10000, 9, 6333, 0, 32)
test_consolidate_partition(Int8, 10000, 9, 6333, 0, 64)
test_consolidate_partition(Int8, 10000, 129, 9999, 0, 16)
test_consolidate_partition(Int8, 10000, 129, 9999, 0, 32)
test_consolidate_partition(Int8, 10000, 129, 9999, 0, 64)
test_consolidate_partition(Int8, 10000, 0, 10000, 1, 16)
test_consolidate_partition(Int8, 10000, 0, 10000, 2, 32)
test_consolidate_partition(Int8, 10000, 0, 10000, 3, 64)
test_consolidate_partition(Int8, 10000, 9, 6333, 4, 16)
test_consolidate_partition(Int8, 10000, 9, 6333, 5, 32)
test_consolidate_partition(Int8, 10000, 9, 6333, 6, 64)
test_consolidate_partition(Int8, 10000, 129, 9999, 7, 16)
test_consolidate_partition(Int8, 10000, 129, 9999, 8, 32)
test_consolidate_partition(Int8, 10000, 129, 9999, 9, 64)
test_consolidate_partition(Int8, 10000, 3329, 9999, 10, 16)
test_consolidate_partition(Int8, 10000, 3329, 9999, 11, 32)
test_consolidate_partition(Int8, 10000, 3329, 9999, 12, 64)

end

function init_case(T, f, N :: Integer)
    a = map(x -> T(f(x)), 1:N)
    c = CuArray(a)
    a, c
end

function init_case(T, f, N :: Tuple)
    a = map(f, rand(N...))
    c = CuArray(a)
    a, c
end

"""
Tests if `c` is a valid sort of `a`
"""
function test_equivalence(a :: Vector, c :: Vector; kwargs...)
    @test counter(a) == counter(c) && issorted(c; kwargs...)
end

"""
Tests if `c` is a valid sort of `a`
"""
function test_equivalence(a :: Array, c :: Array; dims, kwargs...)
    @assert size(a) == size(c)
    nd = ndims(c)
    k = dims
    sz = size(c)

    1 <= k <= nd || throw(ArgumentError("dimension out of range"))

    remdims = ntuple(i -> i == k ? 1 : size(c, i), nd)
    v(a, idx) = view(a, ntuple(i -> i == k ? Colon() : idx[i], nd)...)
    @test all(counter(v(a, idx)) == counter(v(c, idx)) && issorted(v(c, idx); kwargs...) for idx in CartesianIndices(remdims))
end

"""
`T` - Element type to test
`N` - Either an integer for a vector length, or a tuple for array dimension
`f` - For a vector, fill with, for each index i, `T(f(i))`. Facilitates testing orderings
      For an array, fill with `f(rand(T))`. Facilitates testing distributions
"""
function test_sort!(T, N, f=identity; kwargs...)
    original_arr, device_arr = init_case(T, f, N)
    # if sort! fell back to Base implementation, CUDA would error over
    # disallowed scalar indexing
    CUDA.allowscalar(false)
    sort!(device_arr; kwargs...)
    host_result = Array(device_arr)
    test_equivalence(original_arr, host_result; kwargs...)
end

function test_sort(T, N, f=identity; kwargs...)
    original_arr, device_arr = init_case(T, f, N)
    host_result = Array(sort(device_arr; kwargs...))
    test_equivalence(original_arr, host_result; kwargs...)
end

@testset "Quicksort" begin
# test pre-sorted
test_sort!(Int, 1000000)
test_sort!(Int32, 1000000)
test_sort!(Float64, 1000000)
test_sort!(Float32, 1000000)
test_sort!(Int32, 1000000; rev=true)
test_sort!(Float32, 1000000; rev=true)

# test reverse sorted
test_sort!(Int32, 1000000, x -> -x)
test_sort!(Float32, 1000000, x -> -x)
test_sort!(Int32, 1000000, x -> -x; rev=true)
test_sort!(Float32, 1000000, x -> -x; rev=true)

# test random arrays
Random.seed!(0)

test_sort!(Int, 10000, x -> rand(Int))
test_sort!(Int32, 10000, x -> rand(Int32))
test_sort!(Int8, 10000, x -> rand(Int8))
test_sort!(Float64, 10000, x -> rand(Float64))
test_sort!(Float32, 10000, x -> rand(Float32))
test_sort!(Float16, 10000, x -> rand(Float16))

#test non-uniform distributions
test_sort!(UInt8, 100000, x -> round(255 * rand() ^ 2))
test_sort!(UInt8, 100000, x -> round(255 * rand() ^ 3))

# test case when there are more copies of each value than fit in one block
test_sort!(Int8, 4000000, x -> rand(Int8))

# test multiple dimensions
test_sort!(Int32, (4, 50000, 4); dims=2)
test_sort!(Int32, (4, 4, 50000); dims=3, rev=true)

# test various sync depths
CUDA.limit!(CUDA.LIMIT_DEV_RUNTIME_SYNC_DEPTH, 0)
test_sort!(Int, 100000, x -> rand(Int))
CUDA.limit!(CUDA.LIMIT_DEV_RUNTIME_SYNC_DEPTH, 1)
test_sort!(Int, 100000, x -> rand(Int))
CUDA.limit!(CUDA.LIMIT_DEV_RUNTIME_SYNC_DEPTH, 2)
test_sort!(Int, 100000, x -> rand(Int))
CUDA.limit!(CUDA.LIMIT_DEV_RUNTIME_SYNC_DEPTH, 3)
test_sort!(Int, 100000, x -> rand(Int))
CUDA.limit!(CUDA.LIMIT_DEV_RUNTIME_SYNC_DEPTH, 4)
test_sort!(Int, 100000, x -> rand(Int))

# test using a `by` argument
test_sort(Float32, 100000; by=x->abs(x - 0.5))
test_sort(Float64, (4, 100000); by=x->cos(4 * pi * x), dims=2)
end
