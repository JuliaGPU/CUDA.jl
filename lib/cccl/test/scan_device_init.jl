using CUDACore: CuRefArray

@testset "chained scans" begin
    # long enough to span many tiles
    x1 = rand(Int32(1):Int32(100), 100_000)
    x2 = rand(Int32(1):Int32(100), 1_000_000)
    y1 = CCCL.inclusive_scan!(+, CuArray{Int32}(undef, length(x1)), CuArray(x1))
    y2 = CuArray{Int32}(undef, length(x2))
    CCCL.exclusive_scan!(+, y2, CuArray(x2); init=CuRefArray(y1, length(y1)))
    @test Array(y2) == cumsum([x1; 0; x2])[length(x1)+1:end-1]
end

@testset "device references" begin
    for T in (Int64, Float32), n in (1000, 100_000)
        x = T <: Integer ? rand(T(-100):T(100), n) : rand(T, n)
        y = CuArray{T}(undef, length(x))
        CCCL.exclusive_scan!(+, y, CuArray(x); init=CuRef(T(3)))
        @test Array(y) ≈ [3; 3 .+ cumsum(x)[1:end-1]]
        CCCL.exclusive_scan!(max, y, CuArray(x); init=CuRef(T(1)))
        @test Array(y) == accumulate(max, [T(1); x[1:end-1]])
    end

    # nothing to scan, or a single element
    y = CuArray{Int}(undef, 0)
    @test isempty(CCCL.exclusive_scan!(+, y, CuArray{Int}(undef, 0); init=CuRef(1)))
    y = CuArray{Int}(undef, 1)
    @test Array(CCCL.exclusive_scan!(+, y, CuArray([2]); init=CuRef(1))) == [1]

    # device values have no conversion
    @test_throws ArgumentError CCCL.exclusive_scan!(+, y, CuArray([2]); init=CuRef(1f0))
    @test CCCL.unsupported(CCCL.exclusive_scan!, +, y, y; init=CuRef(1f0)) isa String
    @test CCCL.unsupported(CCCL.exclusive_scan!, +, y, y; init=CuRef(1)) === nothing
    @test CCCL.unsupported(CCCL.exclusive_scan!, +, y, y; init=1f0) === nothing

    # CCCL gets inclusive scans with a device init wrong
    @test CCCL.unsupported(CCCL.inclusive_scan!, +, y, y; init=CuRef(1)) isa String
    @test_throws ArgumentError CCCL.inclusive_scan!(+, y, CuArray([2]); init=CuRef(1))
end

@testset "stream order" begin
    # the initial value is read when the scan runs, after the work queued before it
    n = 1_000_000
    x = CuArray(ones(Int, n))
    y = similar(x)
    init = CuArray([0])
    for i in 1:3
        fill!(init, i)
        CCCL.exclusive_scan!(+, y, x; init=CuRefArray(init))
        @test Array(y)[[1, end]] == [i, i + n - 1]
    end
end

@testset "workspace" begin
    x = CuArray(rand(Float32, 10_000))
    y = similar(x)
    init = CuRef(1f0)
    size = CCCL.workspace_size(CCCL.exclusive_scan!, +, y, x; init)
    ws = CuVector{UInt8}(undef, size)
    CCCL.exclusive_scan!(+, y, x; init, workspace=ws)
    @test CUDACore.@allocated(CCCL.exclusive_scan!(+, y, x; init, workspace=ws)) == 0
end

@testset "graph capture" begin
    x = CuArray(rand(Int32(1):Int32(10), 100_000))
    y = similar(x)
    init = CuArray(Int32[0])
    ws = CuVector{UInt8}(undef, CCCL.workspace_size(CCCL.exclusive_scan!, +, y, x;
                                                    init=CuRefArray(init)))
    graph = capture() do
        init .+= Int32(1)
        CCCL.exclusive_scan!(+, y, x; init=CuRefArray(init), workspace=ws)
    end
    exec = instantiate(graph)
    for i in 1:2
        launch(exec)
        @test Array(y) == [0; cumsum(Array(x))[1:end-1]] .+ Int32(i)
    end
end

@testset "host references" begin
    y = CuArray{Int}(undef, 10)
    init = CuRefArray([1])
    @test CCCL.unsupported(CCCL.exclusive_scan!, +, y, y; init) isa String
    @test_throws ArgumentError CCCL.exclusive_scan!(+, y, CuArray(ones(Int, 10)); init)
end
