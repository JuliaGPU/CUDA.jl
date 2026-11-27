struct Particle
    x::Float32
    alive::Bool
end

# the parts of `x` that `three_way_partition!` produces, computed on the CPU
three_way(f1, f2, x) =
    (filter(f1, x), filter(v -> !f1(v) && f2(v), x), filter(v -> !f1(v) && !f2(v), x))

@testset "select!" begin
    for T in (Int32, Float64), n in (1, 2, 1000, 100_001)
        x = rand(T(1):T(100), n)
        d_x = CuArray(x)
        y = CuArray(fill(T(-1), n))
        for f in (v -> rem(v, 3) == 0, <(10), Returns(true), Returns(false))
            k = CCCL.select!(f, y, d_x)
            @test k == count(f, x)
            @test Array(y)[1:k] == filter(f, x)
        end
    end

    # elements past the count are left alone
    x = CuArray(Int32[1, 2, 3, 4])
    y = CuArray(Int32[0, 0, 0, 0, 0, 0])
    @test CCCL.select!(iseven, y, x) == 2
    @test Array(y) == [2, 4, 0, 0, 0, 0]

    # floats are passed as they are
    x = [1.0, NaN, -0.0, 0.0, -NaN, 2.0]
    y = CuArray(zeros(6))
    k = CCCL.select!(isnan, y, CuArray(x))
    @test k == 2 && all(isnan, Array(y)[1:2])
    k = CCCL.select!(signbit, y, CuArray(x))
    @test isequal(Array(y)[1:k], filter(signbit, x))

    # closures, custom element types and arrays of any shape
    particles = [Particle(rand(), rand(Bool)) for _ in 1:10, _ in 1:100]
    d_particles = CuArray(particles)
    y = CuVector{Particle}(undef, length(particles))
    let c = 0.5f0
        f = p -> p.alive & (p.x < c)
        k = CCCL.select!(f, y, d_particles)
        @test Array(y)[1:k] == filter(f, vec(particles))
    end

    # empty input
    @test CCCL.select!(iseven, CuVector{Int}(undef, 0), CuVector{Int}(undef, 0)) == 0

    # counts kept on the device
    x = rand(1:100, 10_000)
    y = CuVector{Int}(undef, length(x))
    for C in (Int32, UInt64), m in (1, 2, 3)
        counts = CuArray(fill(C(42), m))
        @test CCCL.select!(iseven, y, CuArray(x); counts) === nothing
        @test Array(counts) == [count(iseven, x); fill(42, m - 1)]
    end
    counts = CuArray(Int32[42])
    CCCL.select!(iseven, CuVector{Int}(undef, 0), CuVector{Int}(undef, 0); counts)
    @test Array(counts) == [0]

    # invalid arguments
    x = CuArray(rand(1:100, 100))
    @test_throws DimensionMismatch CCCL.select!(iseven, similar(x, 99), x)
    @test_throws ArgumentError CCCL.select!(iseven, similar(x, Int32), x)
    @test_throws ArgumentError CCCL.select!(v -> v % 2, similar(x), x)
    @test_throws ArgumentError CCCL.select!(iseven, x, x)
    let buf = CuVector{Int}(undef, 300)
        @test_throws ArgumentError CCCL.select!(iseven, view(buf, 51:200),
                                                 view(buf, 1:100))
        @test_throws ArgumentError CCCL.select!(iseven, view(buf, 1:100), x;
                                                 counts=view(buf, 100:100))
    end
    @test_throws DimensionMismatch CCCL.select!(iseven, similar(x), x;
                                                counts=CuVector{Int}(undef, 0))
    @test_throws ArgumentError CCCL.select!(iseven, similar(x), x;
                                            counts=CuVector{Float32}(undef, 1))
    let buf = CuVector{UInt8}(undef, 1000)
        misaligned = unsafe_wrap(CuVector{Int}, reinterpret(CuPtr{Int}, pointer(buf, 2)), 100)
        @test_throws ArgumentError CCCL.partition!(iseven, misaligned, x)
    end
end

@testset "filter" begin
    x = randn(Float32, 100_000)
    @test Array(CCCL.filter(>(0), CuArray(x))) == filter(>(0), x)
    x = rand(1:10, 10, 20)
    y = CCCL.filter(iseven, CuArray(x))
    @test y isa CuVector{Int}
    @test Array(y) == filter(iseven, vec(x))
    @test isempty(CCCL.filter(iseven, CuVector{Int}(undef, 0)))
    @test isempty(CCCL.filter(Returns(false), CuArray(x)))
end

@testset "partition!" begin
    for T in (Int8, Int64, Float32), n in (1, 2, 3, 1000, 100_001)
        x = rand(T(1):T(100), n)
        for f in (v -> rem(v, 3) == 0, <(10), Returns(true), Returns(false))
            expected = [filter(f, x); filter(!f, x)]
            y = CuVector{T}(undef, n)
            @test CCCL.partition!(f, y, CuArray(x)) == count(f, x)
            @test Array(y) == expected

            d_x = CuArray(x)
            @test CCCL.partition!(f, d_x) == count(f, x)
            @test Array(d_x) == expected
        end
    end

    # matrices are partitioned in linear order
    x = rand(1:10, 7, 11)
    y = CuVector{Int}(undef, length(x))
    k = CCCL.partition!(isodd, y, CuArray(x))
    @test Array(y) == [filter(isodd, vec(x)); filter(!isodd, vec(x))]

    let c = 3
        x = rand(1:10, 1000)
        d_x = CuArray(x)
        counts = CuArray(Int32[0])
        @test CCCL.partition!(v -> v < c, d_x; counts) === nothing
        @test Array(counts) == [count(<(c), x)]
        @test Array(d_x) == [filter(<(c), x); filter(>=(c), x)]
    end

    @test CCCL.partition!(iseven, CuVector{Int}(undef, 0)) == 0

    x = CuArray(rand(1:100, 100))
    @test_throws DimensionMismatch CCCL.partition!(iseven, similar(x, 101), x)
    @test_throws DimensionMismatch CCCL.workspace_size(CCCL.partition!, iseven,
                                                       similar(x, 101), x)
    @test_throws ArgumentError CCCL.partition!(iseven, x, x)
    @test_throws ArgumentError CCCL.partition!(v -> v % 2, x)
end

@testset "partition" begin
    x = rand(Float32, 10_000)
    yes, no = CCCL.partition(<(0.25f0), CuArray(x))
    @test Array(yes) == filter(<(0.25f0), x)
    @test Array(no) == filter(>=(0.25f0), x)
    yes, no = CCCL.partition(isnan, CuVector{Float32}(undef, 0))
    @test isempty(yes) && isempty(no)
end

@testset "three_way_partition!" begin
    for n in (1, 2, 1000, 100_001)
        x = rand(1:100, n)
        d_x = CuArray(x)
        ys = ntuple(_ -> CuVector{Int}(undef, n), 3)
        # overlapping predicates: the first one takes precedence
        for (f1, f2) in ((<(30), >(60)), (<(50), <(80)), (iseven, Returns(true)))
            parts = three_way(f1, f2, x)
            n1, n2 = CCCL.three_way_partition!(f1, f2, ys..., d_x)
            @test (n1, n2) == length.(parts[1:2])
            @test Array(ys[1])[1:n1] == parts[1]
            @test Array(ys[2])[1:n2] == parts[2]
            @test Array(ys[3])[1:n-n1-n2] == parts[3]
        end
    end

    # discarding parts
    x = rand(1:100, 10_000)
    parts = three_way(<(30), >(60), x)
    y = CuVector{Int}(undef, length(x))
    for i in 1:3
        outputs = ntuple(j -> j == i ? y : nothing, 3)
        counts = CCCL.three_way_partition!(<(30), >(60), outputs..., CuArray(x))
        @test counts == length.(parts[1:2])
        @test Array(y)[1:length(parts[i])] == parts[i]
    end

    let counts = CuArray(Int64[-1, -1, -1])
        @test CCCL.three_way_partition!(<(30), >(60), y, nothing, nothing, CuArray(x);
                                        counts) === nothing
        @test Array(counts) == [length.(parts[1:2])...; -1]
        @test_throws DimensionMismatch CCCL.three_way_partition!(
            <(30), >(60), y, nothing, nothing, CuArray(x); counts=CuArray(Int[0]))
    end

    @test CCCL.three_way_partition!(iseven, isodd, nothing, nothing, nothing,
                                    CuVector{Int}(undef, 0)) == (0, 0)
    @test_throws ArgumentError CCCL.three_way_partition!(<(30), >(60), y, nothing, y,
                                                          CuArray(x))
end

# Run `f(workspace)` with a workspace of the size `f` reports for `nothing`, checking that
# it then allocates nothing, and that a smaller workspace is rejected.
function check_partition_workspace(size, f)
    ws = CuVector{UInt8}(undef, size)
    f(ws)
    @test CUDACore.@allocated(f(ws)) == 0
    @test_throws ArgumentError f(CuVector{UInt8}(undef, size - 1))
end

@testset "workspace" begin
    x = rand(Float32, 100_000)
    src, d_x, y = CuArray(x), CuArray(x), CuArray(x)
    counts = CuVector{Int32}(undef, 2)
    f = <(0.3f0)
    for kwargs in ((;), (; counts=view(counts, 1:1)), (; counts))
        size = CCCL.workspace_size(CCCL.select!, f, y, d_x; kwargs...)
        check_partition_workspace(size, ws -> CCCL.select!(f, y, d_x; kwargs...,
                                                           workspace=ws))

        size = CCCL.workspace_size(CCCL.partition!, f, y, d_x; kwargs...)
        check_partition_workspace(size, ws -> CCCL.partition!(f, y, d_x; kwargs...,
                                                              workspace=ws))
        @test Array(y) == [filter(f, x); filter(!f, x)]

        size = CCCL.workspace_size(CCCL.partition!, f, d_x; kwargs...)
        @test size >= sizeof(x)
        check_partition_workspace(size, ws -> CCCL.partition!(f, copyto!(d_x, src);
                                                              kwargs..., workspace=ws))
        # the rejected workspace left `d_x` as it was
        CCCL.partition!(f, copyto!(d_x, src); kwargs...)
        @test Array(d_x) == [filter(f, x); filter(!f, x)]
    end
    let y2 = similar(y), y3 = similar(y)
        size = CCCL.workspace_size(CCCL.three_way_partition!, f, >(0.6f0), y, y2, y3, d_x;
                                   counts)
        check_partition_workspace(size, ws -> CCCL.three_way_partition!(
            f, >(0.6f0), y, y2, y3, d_x; counts, workspace=ws))
    end
    @test CCCL.workspace_size(CCCL.select!, f, y, CuVector{Float32}(undef, 0)) == 0

    # the workspace must not overlap the arguments
    size = CCCL.workspace_size(CCCL.select!, f, y, d_x)
    let buf = CuVector{UInt8}(undef, size + sizeof(x))
        x′ = reinterpret(Float32, view(buf, 1:sizeof(x)))
        @test_throws ArgumentError CCCL.select!(f, y, x′; workspace=buf)
    end
end

@testset "unsupported" begin
    x = CuArray(rand(Float32, 10))
    y = similar(x)
    counts = CuVector{Int32}(undef, 2)

    builds = length(CCCL.builds)
    @test CCCL.unsupported(CCCL.select!, isnan, y, x) === nothing
    @test CCCL.unsupported(CCCL.select!, isnan, y, CuArray(rand(Float32, 2, 5));
                           counts) === nothing
    @test CCCL.unsupported(CCCL.partition!, v -> v > 0.5f0, y, x) === nothing
    @test CCCL.unsupported(CCCL.partition!, isnan, x) === nothing
    @test CCCL.unsupported(CCCL.three_way_partition!, isnan, signbit, y, nothing, y,
                           x) === nothing
    @test length(CCCL.builds) == builds

    @test CCCL.unsupported(CCCL.select!, isnan, rand(Float32, 10), x) isa String
    @test CCCL.unsupported(CCCL.select!, isnan, nothing, x) isa String
    @test CCCL.unsupported(CCCL.select!, isnan, CuVector{Float64}(undef, 10), x) isa String
    @test CCCL.unsupported(CCCL.select!, isnan, CuArray(rand(Float32, 2, 5)), x) isa String
    @test CCCL.unsupported(CCCL.select!, v -> v + 1, y, x) isa String
    for counts in (CuVector{Int8}(undef, 2), CuArray{Int}(undef, 1, 2), Int[0, 0])
        @test CCCL.unsupported(CCCL.select!, isnan, y, x; counts) isa String
    end
    @test CCCL.unsupported(CCCL.partition!, isnan, CuArray(rand(Float32, 2, 5))) isa String
    let t = CuArray([1f0])
        @test CCCL.unsupported(CCCL.select!, v -> v < t[1], y, x) isa String
    end
end
