# the positions of the first key of each run, comparing every key with its predecessor
run_starts(keys, eq=isequal) =
    [i for i in eachindex(keys) if i == 1 || !eq(keys[i-1], keys[i])]

function check_unique(keys, values=nothing; eq=isequal)
    starts = run_starts(keys, eq)
    d_keys, keys_out = CuArray(keys), CuArray{eltype(keys)}(undef, length(keys))
    if values === nothing
        n = CCCL.unique_consecutive!(keys_out, d_keys; eq)
    else
        values_out = CuArray{eltype(values)}(undef, length(values))
        n = CCCL.unique_by_key!(keys_out, values_out, d_keys, CuArray(values); eq)
        @test Array(values_out)[1:n] == values[starts]
    end
    @test n isa Int
    @test n == length(starts)
    @test isequal(Array(keys_out)[1:n], keys[starts])
    @test isequal(Array(d_keys), keys)
end

@testset "keys $K, values $V" for K in (Int32, Int64, UInt8, Float32), V in (Nothing, Int32)
    for n in (1, 2, 1000, 100_000)
        keys = sort(rand(K <: Integer ? (K(0):K(20)) : K, n))
        values = V === Nothing ? nothing : rand(V, n)
        check_unique(keys, values)
    end
end

@testset "edge cases" begin
    # all equal, all different
    check_unique(fill(Int32(7), 10_000), collect(1:10_000))
    check_unique(fill(Int32(7), 10_000))
    check_unique(collect(Int32, 1:10_000), rand(Float64, 10_000))

    # unsorted keys: only consecutive ones are merged
    keys = Int32[1, 1, 2, 1, 1, 3, 3, 2]
    check_unique(keys, collect(1:8))
    @test Array(CCCL.unique_consecutive(CuArray(keys))) == [1, 2, 1, 3, 2]

    # floats compare by `isequal`, like `unique!` on sorted vectors
    keys = sort([NaN, -NaN, NaN, -0.0, 0.0, 0.0, -Inf, Inf, 1.0, 1.0])
    check_unique(keys, collect(1:10))
    @test isequal(Array(CCCL.unique_consecutive(CuArray(keys))), unique!(copy(keys)))

    # opaque bits types
    keys = [(Int32(i ÷ 3), i % 2 == 0) for i in 0:99]
    check_unique(keys, collect(1:100))
    check_unique(keys)

    # empty input
    keys_out, values_out = CuVector{Int}(undef, 0), CuVector{Int}(undef, 0)
    @test CCCL.unique_by_key!(keys_out, values_out, CuVector{Int}(undef, 0),
                              CuVector{Int}(undef, 0)) === 0
    @test CCCL.unique_consecutive!(keys_out, CuVector{Int}(undef, 0)) === 0
    count = CuArray([42])
    @test CCCL.unique_consecutive!(keys_out, CuVector{Int}(undef, 0); count) === count
    @test Array(count) == [0]
end

@testset "equality operators" begin
    keys = rand(1:10, 10_000)
    check_unique(keys, collect(1:10_000); eq=(a, b) -> a ÷ 3 == b ÷ 3)
    # comparisons are with the predecessor, so runs can drift
    keys = collect(1:10_000)
    check_unique(keys; eq=(a, b) -> b == a + 1)
    let m = 7
        check_unique(sort(rand(1:100, 1000)); eq=(a, b) -> a & ~m == b & ~m)
    end
    check_unique(sort(rand(Float32, 1000)); eq=(a, b) -> abs(a - b) < 0.01f0)
end

@testset "count" begin
    keys = sort(rand(1:100, 10_000))
    starts = run_starts(keys)
    d_keys = CuArray(keys)
    keys_out, values_out = similar(d_keys), CuVector{Int}(undef, length(keys))
    for I in (Int32, Int64, UInt32)
        count = CuVector{I}(undef, 1)
        @test CCCL.unique_consecutive!(keys_out, d_keys; count) === count
        @test Array(count) == [length(starts)]
        count = CuVector{I}(undef, 1)
        @test CCCL.unique_by_key!(keys_out, values_out, d_keys, CuArray(1:10_000);
                                  count) === count
        @test Array(count) == [length(starts)]
        @test Array(values_out)[1:length(starts)] == starts
    end
    @test_throws ArgumentError CCCL.unique_consecutive!(keys_out, d_keys;
                                                        count=CuVector{Int}(undef, 0))
    @test_throws ArgumentError CCCL.unique_consecutive!(keys_out, d_keys;
                                                        count=CuVector{Int8}(undef, 1))
end

@testset "convenience forms" begin
    keys = sort(rand(Int32(1):Int32(50), 1000))
    values = rand(Float32, 1000)
    starts = run_starts(keys)
    ukeys, uvalues = CCCL.unique_by_key(CuArray(keys), CuArray(values))
    @test ukeys isa CuVector{Int32} && uvalues isa CuVector{Float32}
    @test Array(ukeys) == keys[starts]
    @test Array(uvalues) == values[starts]
    @test Array(CCCL.unique_consecutive(CuArray(keys))) == unique(keys)
    @test length(CCCL.unique_consecutive(CuVector{Int}(undef, 0))) == 0
end

@testset "argument checks" begin
    keys, values = CuArray(sort(rand(1:10, 100))), CuArray(rand(100))
    @test_throws DimensionMismatch CCCL.unique_consecutive!(similar(keys, 99), keys)
    @test_throws DimensionMismatch CCCL.unique_by_key!(similar(keys), similar(values),
                                                       keys, values[1:99])
    @test_throws DimensionMismatch CCCL.unique_by_key!(similar(keys), similar(values, 99),
                                                       keys, values)
    @test_throws ArgumentError CCCL.unique_consecutive!(similar(keys, Int32), keys)
    @test_throws ArgumentError CCCL.unique_by_key!(similar(keys), similar(values, Float32),
                                                   keys, values)
    # outputs may not overlap the inputs
    @test_throws ArgumentError CCCL.unique_consecutive!(keys, keys)
    let buf = CuVector{Int}(undef, 150)
        buf[1:100] .= keys
        @test_throws ArgumentError CCCL.unique_consecutive!(view(buf, 51:150),
                                                            view(buf, 1:100))
    end
    @test_throws ArgumentError CCCL.unique_by_key!(similar(keys), keys, keys, values)
    let out = similar(keys)
        @test_throws ArgumentError CCCL.unique_by_key!(out, out, keys, keys)
        @test_throws ArgumentError CCCL.unique_consecutive!(out, keys; count=keys)
        @test_throws ArgumentError CCCL.unique_consecutive!(out, keys; count=view(out, 2:2))
    end
    @test_throws ArgumentError CCCL.unique_consecutive!(similar(keys), keys;
                                                        count=CuVector{Int128}(undef, 1))
end

# with a workspace of the size `f` reports for `nothing`, a call allocates nothing, and a
# smaller workspace is rejected
function check_workspace(size, f)
    ws = CuVector{UInt8}(undef, size)
    f(ws)
    @test CUDACore.@allocated(f(ws)) == 0
    @test_throws ArgumentError f(CuVector{UInt8}(undef, size - 1))
end

@testset "workspace" begin
    n = 100_000
    keys = CuArray(sort(rand(1:1000, n)))
    values = CuArray(rand(Float32, n))
    keys_out, values_out = similar(keys), similar(values)
    count = CuVector{Int32}(undef, 1)
    for kwargs in ((;), (; count), (; eq=(a, b) -> a ÷ 2 == b ÷ 2))
        size = CCCL.workspace_size(CCCL.unique_consecutive!, keys_out, keys; kwargs...)
        check_workspace(size, ws -> CCCL.unique_consecutive!(keys_out, keys; kwargs...,
                                                             workspace=ws))
        size = CCCL.workspace_size(CCCL.unique_by_key!, keys_out, values_out, keys, values;
                                   kwargs...)
        check_workspace(size, ws -> CCCL.unique_by_key!(keys_out, values_out, keys, values;
                                                        kwargs..., workspace=ws))
    end
    @test CCCL.workspace_size(CCCL.unique_consecutive!, CuVector{Int}(undef, 0),
                              CuVector{Int}(undef, 0)) == 0
end

@testset "unsupported" begin
    keys, values = CuArray(rand(1:10, 10)), CuArray(rand(Float32, 10))
    builds = length(CCCL.builds)
    @test CCCL.unsupported(CCCL.unique_consecutive!, similar(keys), keys) === nothing
    @test CCCL.unsupported(CCCL.unique_consecutive!, similar(keys), keys;
                           eq=(a, b) -> a ÷ 2 == b ÷ 2, count=CuVector{Int32}(undef, 1)) ===
          nothing
    @test CCCL.unsupported(CCCL.unique_by_key!, similar(keys), similar(values), keys,
                           values) === nothing
    @test length(CCCL.builds) == builds

    @test CCCL.unsupported(CCCL.unique_consecutive!, similar(keys, Int32), keys) isa String
    @test CCCL.unsupported(CCCL.unique_consecutive!, similar(keys), Array(keys)) isa String
    @test CCCL.unsupported(CCCL.unique_consecutive!, CuArray{Int}(undef, 2, 5),
                           CuArray(rand(1:10, 2, 5))) isa String
    @test CCCL.unsupported(CCCL.unique_consecutive!, similar(keys), keys;
                           count=CuVector{Float32}(undef, 1)) isa String
    @test CCCL.unsupported(CCCL.unique_consecutive!, similar(keys), keys;
                           count=CuVector{Int128}(undef, 1)) isa String
    @test CCCL.unsupported(CCCL.unique_consecutive!, CuVector{Nothing}(undef, 10),
                           CuVector{Nothing}(undef, 10)) isa String
    @test CCCL.unsupported(CCCL.unique_by_key!, similar(keys), similar(values, Float64),
                           keys, values) isa String
    let t = CuArray([1])
        @test CCCL.unsupported(CCCL.unique_consecutive!, similar(keys), keys;
                               eq=(a, b) -> a == b + t[1]) isa String
    end
end
