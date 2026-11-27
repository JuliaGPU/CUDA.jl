using BFloat16s: BFloat16

# segments of sizes CUB sorts differently: with sub-warp, block-wide and radix sorts
function segment_offsets(lengths; start=0)
    offsets = cumsum([start; lengths])
    return offsets
end
const lengths = [0, 1, 2, 3, 0, 7, 31, 100, 257, 1000, 3000, 20_000, 5]

function sort_segments(keys, offsets; values=nothing, rev=false, base=0)
    keys = copy(keys)
    values = values === nothing ? nothing : copy(values)
    for i in 1:length(offsets)-1
        r = offsets[i]+1-base:offsets[i+1]-base
        p = sortperm(keys[r]; rev)
        values === nothing || (values[r] = values[r][p])
        keys[r] = keys[r][p]
    end
    return keys, values
end

function sortperm_segments(keys, offsets; rev=false, base=0)
    ix = collect(1:length(keys))
    for i in 1:length(offsets)-1
        r = offsets[i]+1-base:offsets[i+1]-base
        ix[r] = r[sortperm(keys[r]; rev)]
    end
    return ix
end

sample(T, n) = T == Char ? rand('a':'z', n) : T == Bool ? rand(Bool, n) :
               T <: AbstractFloat ? T.(randn(Float32, n)) : rand(T, n)

# duplicates everywhere, so that stability is observable
few(T, n) = T <: AbstractFloat ? T.(rand(-2:2, n)) : T == Char ? rand('a':'c', n) :
            T == Bool ? rand(Bool, n) : T.(rand(0:3, n))

@testset "$T" for T in (Bool, Char, Int8, UInt16, Int32, Int64, Float16, BFloat16, Float32,
                        Float64)
    offsets = segment_offsets(lengths)
    n = offsets[end]
    for keys in (sample(T, n), few(T, n)), rev in (false, true)
        expected, _ = sort_segments(keys, offsets; rev)
        d_keys = CuArray(keys)
        @test CCCL.segmented_sort!(d_keys, CuArray(offsets); rev) === d_keys
        @test isequal(Array(d_keys), expected)

        values = collect(1:n)
        expected, expected_values = sort_segments(keys, offsets; values, rev)
        d_keys, d_values = CuArray(keys), CuArray(values)
        CCCL.segmented_sort!(d_keys, CuArray(offsets); values=d_values, rev)
        @test isequal(Array(d_keys), expected)
        @test Array(d_values) == expected_values

        @test Array(CCCL.segmented_sortperm(CuArray(keys), CuArray(offsets); rev)) ==
              sortperm_segments(keys, offsets; rev)
    end
end

@testset "floating-point special values" begin
    for T in (Float16, Float32, Float64)
        U = Base.uinttype(T)
        nan_payload = reinterpret(T, reinterpret(U, T(NaN)) | one(U))
        specials = T[1, NaN, -0.0, 0.0, -Inf, -NaN, 2, Inf, -1, -nan_payload, nan_payload]
        offsets = segment_offsets(lengths)
        keys = rand(specials, offsets[end])
        for rev in (false, true)
            d_keys = CuArray(keys)
            CCCL.segmented_sort!(d_keys, CuArray(offsets); rev)
            sorted = Array(d_keys)
            expected, _ = sort_segments(keys, offsets; rev)
            @test isequal(sorted, expected)
            @test sort(reinterpret(U, sorted)) == sort(reinterpret(U, keys))

            # NaNs of any bits are equal, so stay in order
            @test Array(CCCL.segmented_sortperm(CuArray(keys), CuArray(offsets); rev)) ==
                  sortperm_segments(keys, offsets; rev)
        end
    end
end

@testset "offsets" begin
    keys = rand(Float32, 2000)
    values = rand(Int16, 2000)
    for I in (Int32, Int64, UInt32), base in (0, 1)
        # elements before the first and after the last segment stay put
        offsets = I.(segment_offsets([3, 0, 500, 1, 400, 1000]; start=10 + base))
        d_keys, d_values = CuArray(keys), CuArray(values)
        CCCL.segmented_sort!(d_keys, CuArray(offsets); values=d_values, offset_base=base)
        expected, expected_values = sort_segments(keys, offsets; values, base)
        @test Array(d_keys) == expected
        @test Array(d_values) == expected_values
        @test Array(d_keys)[[1:10; 1915:2000]] == keys[[1:10; 1915:2000]]

        ix = CuVector{Int32}(undef, 2000)
        CCCL.segmented_sortperm!(ix, CuArray(keys), CuArray(offsets); offset_base=base)
        @test Array(ix) == sortperm_segments(keys, offsets; base)
    end

    # rows of a CSR matrix, with one-based row pointers
    rowptr = Int32[1, 4, 4, 10, 11]
    colval = Int32[3, 1, 2, 9, 4, 1, 8, 2, 5, 7]
    nzval = Float64.(1:10)
    d_colval, d_nzval = CuArray(colval), CuArray(nzval)
    CCCL.segmented_sort!(d_colval, CuArray(rowptr); values=d_nzval, offset_base=1)
    @test Array(d_colval) == Int32[1, 2, 3, 1, 2, 4, 5, 8, 9, 7]
    @test Array(d_nzval) == [2, 3, 1, 6, 8, 5, 9, 7, 4, 10]

    # equal-length segments, like sorting the columns of a matrix
    A = rand(Int32, 300, 50)
    d_A = CuArray(A)
    CCCL.segmented_sort!(d_A, CuArray(0:300:length(A)))
    @test Array(d_A) == sort(A; dims=1)
    @test Array(CCCL.segmented_sortperm(CuArray(A), CuArray(0:300:length(A)))) ==
          sortperm(A; dims=1)

    @test_throws ArgumentError CCCL.segmented_sort!(CuArray(keys), CuArray([0, 2000]);
                                                    offset_base=2)
    @test_throws DimensionMismatch CCCL.segmented_sort!(CuArray(keys), CuArray([0, 2000]);
                                                        values=CuArray(values[1:10]))
end

@testset "degenerate" begin
    # nothing to sort, and no build that would be reused for sorts of pairs
    for keys in (Int16[], Int16[7]), offsets in ([0], [0, length(keys)])
        d_keys = CuArray(keys)
        CCCL.segmented_sort!(d_keys, CuArray(offsets); values=CuArray(Int8.(keys)))
        @test Array(d_keys) == keys
        CCCL.segmented_sort!(d_keys, CuArray(offsets))
        @test Array(d_keys) == keys
        @test Array(CCCL.segmented_sortperm(d_keys, CuArray(offsets))) == 1:length(keys)
    end
    @test Array(CCCL.segmented_sortperm(CuArray(rand(10)), CuArray(Int[]))) == 1:10
    keys = rand(Int16, 1000)
    values = rand(Int16, 1000)
    d_keys, d_values = CuArray(keys), CuArray(values)
    CCCL.segmented_sort!(d_keys, CuArray([0, 1000]); values=d_values)
    @test Array(d_values) == values[sortperm(keys)]

    # all keys equal
    d_keys, d_values = CuArray(fill(1f0, 1000)), CuArray(collect(1:1000))
    CCCL.segmented_sort!(d_keys, CuArray([0, 500, 1000]); values=d_values, rev=true)
    @test Array(d_values) == 1:1000
end

@testset "workspace" begin
    n = 100_000
    keys = rand(Float32, n)
    values = rand(Int32, n)
    offsets = CuArray(Int32[0, 10, 1000, 30_000, n])
    src, d_keys, d_values = CuArray(keys), CuArray(keys), CuArray(values)
    for (offsets, kwargs) in ((offsets, (;)), (offsets, (; values=d_values)),
                              (offsets .+ Int32(1), (; rev=true, offset_base=1)))
        size = CCCL.workspace_size(CCCL.segmented_sort!, d_keys, offsets; kwargs...)
        ws = CuVector{UInt8}(undef, size)
        f() = CCCL.segmented_sort!(copyto!(d_keys, src), offsets; kwargs..., workspace=ws)
        f()
        @test CUDACore.@allocated(f()) == 0
        rev = get(kwargs, :rev, false)
        @test Array(d_keys) == sort_segments(keys, Array(offsets); rev,
                                             base=get(kwargs, :offset_base, 0))[1]
        small = CuVector{UInt8}(undef, size - 1)
        @test_throws ArgumentError CCCL.segmented_sort!(d_keys, offsets; kwargs...,
                                                        workspace=small)
    end
    for I in (Int32, Int64)
        ix = CuVector{I}(undef, n)
        size = CCCL.workspace_size(CCCL.segmented_sortperm!, ix, src, offsets)
        ws = CuVector{UInt8}(undef, size)
        CCCL.segmented_sortperm!(ix, src, offsets; workspace=ws)
        @test CUDACore.@allocated(CCCL.segmented_sortperm!(ix, src, offsets;
                                                            workspace=ws)) == 0
        @test Array(ix) == sortperm_segments(keys, Array(offsets))
    end
    @test CCCL.workspace_size(CCCL.segmented_sort!, CuArray(Float32[1]), offsets) == 0
end

@testset "unsupported" begin
    x = CuArray(rand(Float32, 10))
    offsets = CuArray([0, 10])
    builds = length(CCCL.builds)
    @test CCCL.unsupported(CCCL.segmented_sort!, x, offsets) === nothing
    @test CCCL.unsupported(CCCL.segmented_sort!, x, offsets; values=CuArray(rand(Int8, 10)),
                           rev=true, offset_base=1) === nothing
    @test CCCL.unsupported(CCCL.segmented_sortperm!, CuVector{Int}(undef, 10),
                           CuArray(rand(Char, 10)), offsets) === nothing
    @test length(CCCL.builds) == builds

    @test CCCL.unsupported(CCCL.segmented_sort!, CuArray(rand(Int128, 10)),
                           offsets) isa String
    @test CCCL.unsupported(CCCL.segmented_sort!, x, CuArray(Float32[0, 10])) isa String
    @test CCCL.unsupported(CCCL.segmented_sort!, x, CuArray(Int128[0, 10])) isa String
    @test CCCL.unsupported(CCCL.segmented_sort!, x, [0, 10]) isa String
    @test CCCL.unsupported(CCCL.segmented_sort!, x, offsets; values=rand(10)) isa String
    @test CCCL.unsupported(CCCL.segmented_sortperm!, similar(x), x, offsets) isa String
end
