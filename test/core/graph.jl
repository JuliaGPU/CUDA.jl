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

@testset "empty graphs" begin
    @test length(CuGraph()) == 0
    @test isempty(CUDA.nodes(capture(() -> nothing)))
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

    # a capture that fails to begin leaves the capture in progress alone
    graph = capture() do
        @test_throws CuError capture(() -> nothing)
        a .+= 1
    end
    launch(instantiate(graph))
    @test Array(a) == fill(2, 4)
    a .= 1

    # the stream is still usable
    a .+= 1
    @test Array(a) == fill(2, 4)
end

@testset "unrelated tasks during capture" begin
    a = CUDA.zeros(Float32, 16)
    a .+= 1
    synchronize()

    # tasks that run on the capturing thread while the capturing task waits
    ops = [() -> CUDA.zeros(Float32, 1024),
           () -> synchronize(),
           () -> Array(CUDA.ones(Float32, 16)),
           () -> CUDA.unsafe_free!(CUDA.zeros(Float32, 1024))]
    for op in ops, yielder in (:sleep, :lock)
        a .= 1
        lk = ReentrantLock()
        started = Base.Event()
        other = @async begin
            lock(lk) do
                notify(started)
                yield()
                op()
            end
        end
        wait(started)
        graph = capture() do
            a .+= 1
            if yielder === :sleep
                sleep(0.01)
                wait(other)
            else
                # a contended lock, like the lock of memory that another task is using
                lock(() -> nothing, lk)
            end
            a .+= 1
        end
        @test (fetch(other); true)
        instantiate(graph)()
        @test Array(a) == fill(3f0, 16)
    end
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
