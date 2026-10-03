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

@testset "memory is kept alive" begin
    x = CUDA.ones(Float32, 1024)
    y = CUDA.zeros(Float32, 1024)
    y .+= x
    exec = instantiate(capture(() -> y .+= x))

    # freeing the input, and allocating new memory that could reuse it, doesn't affect the
    # graph. the executable graph also doesn't need the graph to be kept around.
    CUDA.unsafe_free!(x)
    x = nothing
    GC.gc(true)
    z = CUDA.fill(42f0, 1024)
    exec()
    exec()
    @test Array(y) == fill(3f0, 1024)
    @test Array(z) == fill(42f0, 1024)

    # neither do views or wrapped arrays
    p = CUDA.ones(Float32, 2048)
    v = view(p, 1025:2048)
    w = unsafe_wrap(CuArray, fill(2f0, 1024))
    kernel(y, v, w) = (i = threadIdx().x; y[i] += v[i] + w[i]; nothing)
    @cuda threads=1024 kernel(y, v, w)
    exec = instantiate(capture(() -> @cuda threads=1024 kernel(y, v, w)))
    p = v = w = nothing
    GC.gc(true)
    CUDA.fill(0f0, 2048)
    exec()
    @test Array(y) == fill(9f0, 1024)
end

@testset "allocations" begin
    x = CUDA.ones(Float32, 1024)
    y = x .+ 1
    local z
    exec = instantiate(capture() do
        z = x .+ 1          # memory is allocated outside of the graph
        y .= 2 .* z
    end)
    @test size(z) == (1024,)

    used = CUDA.used_memory()
    for i in 1:10
        x .= i
        exec()
        # the allocated memory is reused, and overwritten by every launch
        @test Array(z) == fill(Float32(i + 1), 1024)
        @test Array(y) == fill(Float32(2(i + 1)), 1024)
    end
    @test CUDA.used_memory() == used

    # memory allocated during capture can be used right away
    local w
    graph = capture(() -> w = CUDA.zeros(Float32, 4))
    copyto!(w, Float32[1, 2, 3, 4])
    @test Array(w) == [1, 2, 3, 4]
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

    # copying to host memory that the graph can't keep alive
    pinned = CUDA.pin(zeros(Int, 4))
    @test_throws CaptureError capture(() -> copyto!(pinned, a))

    # other errors are always reported
    @test_throws ArgumentError capture(() -> throw(ArgumentError("oops")); throw_error=false)
    @test !is_capturing()

    # arrays from an allocation cache would be reused when the cache's scope ends
    cache = GPUArrays.AllocCache()
    GPUArrays.@cached cache begin
        b = CUDA.zeros(Int, 4)
        @test_throws CaptureError capture(() -> b .+= 1)
    end

    # the stream is still usable
    a .+= 1
    @test Array(a) == fill(2, 4)
end

@testset "copies from the CPU" begin
    a = CUDA.zeros(Float32, 4)
    h = Float32[1, 2, 3, 4]
    exec = instantiate(capture(() -> copyto!(a, h)))

    # data is copied when capturing
    h .= 0
    exec()
    @test Array(a) == [1, 2, 3, 4]

    # also from pinned memory
    h = CUDA.pin(Float32[5, 6, 7, 8])
    exec = instantiate(capture(() -> copyto!(a, h)))
    h .= 0
    exec()
    @test Array(a) == [5, 6, 7, 8]

    # to use host memory when launching a graph, use arrays backed by host memory
    h = CuArray{Float32,1,CUDA.HostMemory}([1, 2, 3, 4])
    exec = instantiate(capture(() -> copyto!(a, h)))
    h .= 9
    exec()
    @test Array(a) == fill(9, 4)
    exec = instantiate(capture(() -> copyto!(h, a .+ 1)))
    exec()
    synchronize()
    @test Array(h) == fill(10, 4)
end

@testset "multitasking" begin
    a = CUDA.zeros(Float32, 1024)
    b = CUDA.zeros(Float32, 1024)
    a .+= 1
    exec = instantiate(capture(() -> a .+= 1))

    # launching a graph synchronizes with other tasks like other operations do
    for i in 1:10
        @sync begin
            Threads.@spawn begin
                exec()
            end
        end
        b .+= a
    end
    @test Array(a) == fill(11f0, 1024)
    @test Array(b) == fill(sum(2:11), 1024)
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
