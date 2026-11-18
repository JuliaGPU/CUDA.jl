using CCCL, CUDACore, Test

@test CCCL.functional()

@testset "reduce" begin
    for T in (Int32, Int64, Float32, Float64)
        x = rand(T(1):T(100), 10_000)
        @test CCCL.reduce(+, CuArray(x); init=zero(T)) == sum(x)
        @test CCCL.reduce(max, CuArray(x); init=typemin(T)) == maximum(x)
    end

    # arbitrary Julia functions
    x = rand(Float32, 10_000)
    @test CCCL.reduce((a, b) -> max(abs(a), abs(b)), CuArray(x .- 0.5f0); init=0f0) ≈
          maximum(abs, x .- 0.5f0)

    # custom element types
    x = [ComplexF32(rand(), rand()) for _ in 1:1000]
    @test CCCL.reduce(+, CuArray(x); init=zero(ComplexF32)) ≈ sum(x)
end

@testset "scan" begin
    x = rand(Int32(1):Int32(100), 10_000)
    y = CuArray{Int32}(undef, length(x))
    CCCL.inclusive_scan!(+, y, CuArray(x))
    @test Array(y) == cumsum(x)
    CCCL.inclusive_scan!(+, y, CuArray(x); init=Int32(10))
    @test Array(y) == cumsum(x) .+ 10
    CCCL.exclusive_scan!(+, y, CuArray(x); init=Int32(0))
    @test Array(y) == [0; cumsum(x)[1:end-1]]
    CCCL.inclusive_scan!(max, y, CuArray(x))
    @test Array(y) == accumulate(max, x)
end

@testset "segmented reduce" begin
    x = rand(Float32, 1000)
    offsets = [0, 3, 3, 10, 1000]
    out = CuArray{Float32}(undef, length(offsets) - 1)
    CCCL.segmented_reduce!(+, out, CuArray(x), CuArray(offsets); init=0f0)
    @test Array(out) ≈ [sum(x[offsets[i]+1:offsets[i+1]]; init=0f0) for i in 1:length(offsets)-1]
end

@testset "radix sort" begin
    for T in (UInt8, Int32, UInt64, Float32, Float64)
        keys = rand(T, 10_000)
        d_keys = CuArray(keys)
        CCCL.radix_sort!(d_keys)
        @test Array(d_keys) == sort(keys)

        d_keys = CuArray(keys)
        d_values = CuArray(collect(Int32(1):Int32(length(keys))))
        CCCL.radix_sort!(d_keys; values=d_values, rev=true)
        @test Array(d_keys) == sort(keys; rev=true)
        @test keys[Array(d_values)] == Array(d_keys)
    end
end

@testset "merge sort" begin
    keys = rand(Float64, 10_000)
    d_keys = CuArray(keys)
    CCCL.merge_sort!(d_keys)
    @test Array(d_keys) == sort(keys)

    # comparators capturing state
    let c = 0.5
        d_keys = CuArray(keys)
        CCCL.merge_sort!(d_keys; lt=(a, b) -> abs(a - c) < abs(b - c))
        @test Array(d_keys) == sort(keys; by=k -> abs(k - c))
    end

    d_keys = CuArray(keys)
    d_values = CuArray(collect(1:length(keys)))
    CCCL.merge_sort!(d_keys; lt=(>), values=d_values)
    @test Array(d_keys) == sort(keys; rev=true)
    @test keys[Array(d_values)] == Array(d_keys)
end
