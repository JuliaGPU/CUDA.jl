using Random
using DataStructures

function init_case(T, f, N::Integer)
    a = map(x -> T(f(x)), 1:N)
    c = CuArray(a)
    a, c
end

function init_case(T, f, N::Tuple)
    a = map(f, rand(T, N...))
    c = CuArray(a)
    a, c
end

"""
Tests if `c` is a valid sort of `a`
"""
function check_equivalence(a::Vector, c::Vector; kwargs...)
    counter(a) == counter(c) && issorted(c; kwargs...)
end

"""
Tests if `c` is a valid sort of `a`
"""
function check_equivalence(a::Array, c::Array; dims, kwargs...)
    @assert size(a) == size(c)
    nd = ndims(c)
    k = dims
    sz = size(c)

    1 <= k <= nd || throw(ArgumentError("dimension out of range"))

    remdims = ntuple(i -> i == k ? 1 : size(c, i), nd)
    v(a, idx) = view(a, ntuple(i -> i == k ? Colon() : idx[i], nd)...)
    all(counter(v(a, idx)) == counter(v(c, idx)) && issorted(v(c, idx); kwargs...)
        for idx in CartesianIndices(remdims))
end

"""
`T` - Element type to test
`N` - Either an integer for a vector length, or a tuple for array dimension
`f` - For a vector, fill with, for each index i, `T(f(i))`. Facilitates testing orderings
      For an array, fill with `f(rand(T))`. Facilitates testing distributions
"""
function check_sort!(T, N, f=identity; kwargs...)
    original_arr, device_arr = init_case(T, f, N)
    sort!(device_arr; kwargs...)
    host_result = Array(device_arr)
    check_equivalence(original_arr, host_result; kwargs...)
end

function check_sort(T, N, f=identity; kwargs...)
    original_arr, device_arr = init_case(T, f, N)
    host_result = Array(sort(device_arr; kwargs...))
    check_equivalence(original_arr, host_result; kwargs...)
end

"""
Tests if `c` is a valid sort of `a`
"""
function check_partial_equivalence(a::Vector, c::Vector, partial_k; kwargs...)
    # check that the right amount of elements are present
    if counter(a) != counter(c)
        return false
    end
    # check that the range partial_k is sorted
    if !issorted(c; kwargs...)
        return false
    end
    lo, hi = first(partial_k), last(partial_k)
    # check that everything left of partial_k is lesser, and everything right greater
    if :by in keys(kwargs)
        c = map(kwargs[:by], c)
    end
    if ! all(x <= c[lo] for x in c[1:lo]) || !all(x >= c[hi] for x in c[hi:end])
    end
    return true
end

function check_partialsort!(T, N, partial_k, f=identity; kwargs...)
    original_arr, device_arr = init_case(T, f, N)
    out = partialsort!(device_arr, partial_k; kwargs...)
    right_size = size(out) == size(partial_k)
    host_result = Array(device_arr)
    right_size && check_partial_equivalence(original_arr, host_result, partial_k; kwargs...)
end

function check_sortperm!(i, T, N; kwargs...)
    I = CuArray(i)
    a = rand(T, N)
    c = CuArray(a)
    sortperm!(I, c; kwargs...)
    return Array(I) == sortperm!(i, a; kwargs...)
end


function check_sortperm(T, N; kwargs...)
    a = rand(T, N)
    c = CuArray(a)
    I = sortperm(c; kwargs...)
    return Array(I) == sortperm(a; kwargs...)
end

@testset "interface" begin
    @testset "sort" begin
        # pre-sorted
        @test check_sort!(Int, 1000000)
        @test check_sort!(Int32, 1000000)
        @test check_sort!(Float64, 1000000)
        @test check_sort!(Float32, 1000000)
        @test check_sort!(Int32, 1000000; rev=true)
        @test check_sort!(Float32, 1000000; rev=true)

        # reverse sorted
        @test check_sort!(Int32, 1000000, x -> -x)
        @test check_sort!(Float32, 1000000, x -> -x)
        @test check_sort!(Int32, 1000000, x -> -x; rev=true)
        @test check_sort!(Float32, 1000000, x -> -x; rev=true)

        @test check_sort!(Int, 10000, x -> rand(Int))
        @test check_sort!(Int32, 10000, x -> rand(Int32))
        @test check_sort!(Int8, 10000, x -> rand(Int8))
        @test check_sort!(Float64, 10000, x -> rand(Float64))
        @test check_sort!(Float32, 10000, x -> rand(Float32))
        @test check_sort!(Float16, 10000, x -> rand(Float16))
        @test check_sort!(Tuple{Int,Int}, 10000, x -> (rand(Int), rand(Int)))

        # non-uniform distributions
        @test check_sort!(UInt8, 100000, x -> round(255 * rand() ^ 2))
        @test check_sort!(UInt8, 100000, x -> round(255 * rand() ^ 3))

        # more copies of each value than can fit in one block
        @test check_sort!(Int8, 4000000, x -> rand(Int8))

        # multiple dimensions
        @test check_sort!(Int32, (4, 50000, 4); dims=2)
        @test check_sort!(Int32, (2, 2, 50000); dims=3, rev=true)

        # large sizes
        @test check_sort!(Float32, 2^22)

        # using a `by` argument
        @test check_sort(Float32, 100000; by=x->abs(x - 0.5))
        @test check_sort!(Float32, (100000, 4); by=x->abs(x - 0.5), dims=1)
        @test check_sort!(Float32, (4, 100000); by=x->abs(x - 0.5), dims=2)
        @test check_sort!(Float64, 400000; by=x->8*x-round(8*x))
        @test check_sort!(Float64, (100000, 4); by=x->8*x-round(8*x), dims=1)
        @test check_sort!(Float64, (4, 100000); by=x->8*x-round(8*x), dims=2)
        # small inputs with many duplicates
        @test check_sort!(Int, 200; by=x->x % 2)
        @test check_sort!(Int, 200; by=x->x % 3)
        @test check_sort!(Int, 200; by=x->x % 4)

        # out of place
        @test check_sort(Int, 10000, x -> rand(Int))

        # sizes around block boundaries
        @test check_sort!(Float32, 1, x -> rand(Float32))
        @test check_sort!(Float32, 2, x -> rand(Float32))
        @test check_sort!(Float32, 3, x -> rand(Float32))
        @test check_sort!(Float32, 4, x -> rand(Float32))
        @test check_sort!(Float32, 1 << 16 + 0, x -> rand(Float32))
        @test check_sort!(Float32, 1 << 16 + 1, x -> rand(Float32))
        @test check_sort!(Float32, 1 << 16 + 31, x -> rand(Float32))
        @test check_sort!(Float32, 1 << 16 + 32, x -> rand(Float32))
        @test check_sort!(Float32, 1 << 16 + 33, x -> rand(Float32))
        @test check_sort!(Float32, 1 << 16 + 127, x -> rand(Float32))
        @test check_sort!(Float32, 1 << 16 + 128, x -> rand(Float32))
        @test check_sort!(Float32, 1 << 16 + 129, x -> rand(Float32))
    end

    #partial sort
    @test check_partialsort!(Int, 100000, 1)
    @test check_partialsort!(Int, 100000, 100000)
    @test check_partialsort!(Int, 100000, 50000)
    @test check_partialsort!(Int, 100000, 10000:20000)
    @test check_partialsort!(Int, 100000, 1:100000)
    @test check_partialsort!(Float32, 100000, 1; by=x->abs(x - 0.5))
    @test check_partialsort!(Float32, 100000, 100000; by=x->abs(x - 0.5))
    @test check_partialsort!(Float32, 100000, 50000; by=x->abs(x - 0.5))
    @test check_partialsort!(Float32, 100000, 10000:20000; by=x->abs(x - 0.5))
    @test check_partialsort!(Float32, 100000, 1:100000; by=x->abs(x - 0.5))

    #sort perm
    # A set of 1e6 Float32s has ~9.4e5 unique values: stability is non-trivial
    @test check_sortperm(Float32, 0)
    @test check_sortperm(Float32, 1000000)
    @test check_sortperm(Float32, 1000000; rev=true)
    @test check_sortperm(Float32, 1000000; by=x->abs(x-0.5f0))
    @test check_sortperm(Float32, 1000000; rev=true, by=x->abs(x-0.5f0))
    @test check_sortperm(Float64, 1000000)
    @test check_sortperm(Float64, 1000000; rev=true)
    @test check_sortperm(Float64, 1000000; by=x->abs(x-0.5))
    @test check_sortperm(Float64, 1000000; rev=true, by=x->abs(x-0.5))
    @test check_sortperm(Float32, (100_000, 16); dims=1)
    @test check_sortperm(Float32, (100_000, 16); dims=2)
    @test check_sortperm(Float32, (100, 256, 256); dims=1)

    # check with Int32 indices
    @test check_sortperm!(collect(Int32(1):Int32(1000000)), Float32, 1000000)
    @test check_sortperm!(collect(Int32(1):Int32(0)), Float32, 0)
    # `initialized` kwarg
    @test check_sortperm!(collect(Int32(1):Int32(1000000)), Float32, 1000000; initialized=true)
    @test check_sortperm!(collect(Int32(1):Int32(1000000)), Float32, 1000000; initialized=false)
    # expected error case
    @test_throws ArgumentError sortperm!(CuArray(1:3), CuArray(1:4))
    # mismatched types (JuliaGPU/CUDA.jl#2046)
    @test check_sortperm!(collect(UInt64(1):UInt64(1000000)), Int64, 1000000)
end
