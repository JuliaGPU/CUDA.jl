using Random

@testset "capture and launch" begin
    a = CUDA.zeros(Int, 4)
    a .+= 1     # compile outside of the capture
    @test !is_capturing()

    graph = capture() do
        @test is_capturing()
        a .+= 1
    end
    @test !is_capturing()
    @test Array(a) == fill(1, 4)    # capturing doesn't execute anything

    exec = instantiate(graph)
    launch(exec)
    @test Array(a) == fill(2, 4)
    exec()
    exec()
    @test Array(a) == fill(4, 4)

    @test sprint(show, MIME"text/plain"(), graph) == "CuGraph with 1 node: 1 kernel"
    @test occursin("digraph", sprint(show, MIME"text/vnd.graphviz"(), graph))

    # the first launch of a graph can be made faster by uploading it
    exec = instantiate(graph)
    upload(exec)
    exec()
    @test Array(a) == fill(5, 4)
end

@testset "update" begin
    a = CUDA.zeros(Int, 4)
    b = CUDA.zeros(Int, 4)
    a .+= 1
    exec = instantiate(capture(() -> a .+= 1))

    @test update(exec, capture(() -> b .+= 2))
    exec()
    @test Array(a) == fill(1, 4)
    @test Array(b) == fill(2, 4)

    # a graph with different operations can't be used
    @test !update(exec, capture(() -> (b .+= 1; b .+= 1)); throw_error=false)
    @test_throws ErrorException update(exec, capture(() -> (b .+= 1; b .+= 1)))
end

@testset "garbage collection during capture" begin
    # garbage that was last used on various streams, using various kinds of memory
    function garbage(M, s)
        stream!(s) do
            fill!(CuArray{Float32,1,M}(undef, 1024), 1)
        end
        return
    end

    x = CUDA.zeros(Float32, 1)
    x .+= 1
    for M in (CUDA.DeviceMemory, CUDA.UnifiedMemory, CUDA.HostMemory),
        s in (stream(), CuStream())
        garbage(M, s)
        synchronize(s)
        graph = capture() do
            x .+= 1
            GC.gc(true)
            x .+= 1
        end
        exec = instantiate(graph)
        exec()
        @test Array(x) == [3f0]
        x .= 1
    end
end

@testset "unsupported operations" begin
    a = CUDA.zeros(Int, 4)
    a .+= 1

    # waiting for the GPU
    @test_throws CaptureError capture(() -> Array(a))
    @test_throws CaptureError capture(() -> synchronize())
    @test_throws CaptureError capture(() -> device_synchronize())
    @test_throws CaptureError capture(() -> copyto!(zeros(Int, 4), a))
    @test !is_capturing()

    # those failures can be ignored
    @test capture(() -> Array(a); throw_error=false) === nothing
    @test !is_capturing()
    event = CuEvent()
    @test capture(() -> (record(event); synchronize(event)); throw_error=false) === nothing
    @test !is_capturing()

    # other errors are always reported
    @test_throws ArgumentError capture(() -> throw(ArgumentError("oops")); throw_error=false)
    @test !is_capturing()

    # the stream is still usable
    a .+= 1
    @test Array(a) == fill(2, 4)
end

@testset "@captured" begin
    a = CUDA.zeros(Int, 1)
    function iteration(a, val)
        # custom kernel to force compilation on the first iteration
        function kernel(a, val)
            a[] += val
            return
        end
        @cuda kernel(a, val)
        return
    end

    for i in 1:3
        @captured iteration(a, i)
    end
    @test Array(a) == [6]
end
