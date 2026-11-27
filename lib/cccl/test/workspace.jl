# scratch memory from a workspace, and what CCCL supports

# Run `f(workspace)` with a workspace of the size `f` reports for `nothing`, checking that
# it then allocates nothing, and that a smaller workspace is rejected.
function check_workspace(size, f)
    ws = CuVector{UInt8}(undef, size)
    f(ws)
    @test CUDACore.@allocated(f(ws)) == 0
    @test_throws ArgumentError f(CuVector{UInt8}(undef, size - 1))
end

@testset "workspace" begin
    n = 100_000
    x = CuArray(rand(Float32, n))
    y = similar(x)
    out = CuArray{Float32}(undef, 1)

    size = CCCL.workspace_size(CCCL.reduce!, +, out, x; init=0f0)
    check_workspace(size, ws -> CCCL.reduce!(+, out, x; init=0f0, workspace=ws))
    @test Array(out)[] ≈ sum(Array(x))

    for (f, kwargs) in ((CCCL.inclusive_scan!, (;)), (CCCL.exclusive_scan!, (; init=0f0)))
        size = CCCL.workspace_size(f, +, y, x; kwargs...)
        check_workspace(size, ws -> f(+, y, x; kwargs..., workspace=ws))
    end
    @test Array(y) ≈ [0; cumsum(Array(x))[1:end-1]]

    offsets = CuArray(Int32[0, 10, 1000, n])
    sums = CuArray{Float32}(undef, 3)
    size = CCCL.workspace_size(CCCL.segmented_reduce!, +, sums, x, offsets; init=0f0)
    check_workspace(size, ws -> CCCL.segmented_reduce!(+, sums, x, offsets; init=0f0,
                                                        workspace=ws))

    # radix sorts need buffers for the keys and values too
    keys = rand(Float32, n)
    values = rand(Int32, n)
    src, d_keys = CuArray(keys), CuArray(keys)
    for kwargs in ((;), (; values=CuArray(values)), (; rev=true))
        size = CCCL.workspace_size(CCCL.radix_sort!, d_keys; kwargs...)
        @test size >= n * sizeof(Float32)
        check_workspace(size, ws -> CCCL.radix_sort!(copyto!(d_keys, src); kwargs...,
                                                     workspace=ws))
    end
    let d_keys = CuArray(keys), d_values = CuArray(values)
        size = CCCL.workspace_size(CCCL.radix_sort!, d_keys; values=d_values)
        CCCL.radix_sort!(d_keys; values=d_values,
                         workspace=CuVector{UInt8}(undef, size))
        @test Array(d_keys) == sort(keys)
        @test Array(d_values) == values[sortperm(keys)]
    end

    for I in (Int32, Int64)
        ix = CuVector{I}(undef, n)
        d_keys = CuArray(keys)
        size = CCCL.workspace_size(CCCL.radix_sortperm!, ix, d_keys)
        check_workspace(size, ws -> CCCL.radix_sortperm!(ix, d_keys; workspace=ws))
        @test Array(ix) == sortperm(keys)
    end

    let d_keys = CuArray(keys), d_values = CuArray(values)
        lt = (a, b) -> a > b
        size = CCCL.workspace_size(CCCL.merge_sort!, d_keys; lt, values=d_values)
        check_workspace(size, ws -> CCCL.merge_sort!(d_keys; lt, values=d_values,
                                                     workspace=ws))
        @test Array(d_keys) == sort(keys; rev=true)
    end

    # nothing to sort, nothing needed
    @test CCCL.workspace_size(CCCL.radix_sort!, CuArray(Float32[1])) == 0
    @test CCCL.workspace_size(CCCL.radix_sortperm!, CuVector{Int}(undef, 0),
                              CuVector{Int8}(undef, 0)) == 0

    # workspaces need to be aligned like CUB's temporary storage
    size = CCCL.workspace_size(CCCL.radix_sort!, CuArray(keys))
    ws = CuVector{UInt8}(undef, size + 1)
    @test_throws ArgumentError CCCL.radix_sort!(CuArray(keys); workspace=view(ws, 2:size+1))
end

@testset "unsupported" begin
    x = CuArray(rand(Float32, 10))
    ix = CuVector{Int}(undef, 10)

    builds = length(CCCL.builds)
    @test CCCL.unsupported(CCCL.radix_sort!, CuArray(rand(UInt16, 10))) === nothing
    @test CCCL.unsupported(CCCL.radix_sort!, x; values=CuArray(rand(Int8, 10)),
                           rev=true) === nothing
    @test CCCL.unsupported(CCCL.radix_sortperm!, ix, CuArray(rand(Char, 10))) === nothing
    @test CCCL.unsupported(CCCL.merge_sort!, x; lt=(a, b) -> a > b) === nothing
    @test CCCL.unsupported(CCCL.reduce!, max, similar(x, 1), x; init=0f0) === nothing
    @test CCCL.unsupported(CCCL.inclusive_scan!, *, similar(x), x) === nothing
    @test CCCL.unsupported(CCCL.segmented_reduce!, +, x, x, CuArray([0, 10]);
                           init=0f0) === nothing
    # deciding never builds anything
    @test length(CCCL.builds) == builds

    @test CCCL.unsupported(CCCL.radix_sort!, CuArray(rand(Int128, 10))) isa String
    @test CCCL.unsupported(CCCL.radix_sort!, rand(Float32, 10)) isa String
    @test CCCL.unsupported(CCCL.radix_sort!, view(x, 1:2:10)) isa String
    @test CCCL.unsupported(CCCL.radix_sort!, x; values=CuArray(rand(10))') isa String
    @test CCCL.unsupported(CCCL.radix_sortperm!, CuVector{Float32}(undef, 10), x) isa String
    @test CCCL.unsupported(CCCL.radix_sortperm!, ix, CuArray(rand(Float32, 2, 5))) isa String
    @test CCCL.unsupported(CCCL.reduce!, +, CuVector{Float64}(undef, 1), x; init=0) isa String
    let t = CuArray([1f0])
        # a closure capturing an array
        @test CCCL.unsupported(CCCL.merge_sort!, x; lt=(a, b) -> a < b + t[1]) isa String
    end
    @test CCCL.unsupported(sort!, x) isa String
end
