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

    # free the garbage of earlier tests first, as collecting it during the loop would lower
    # the memory usage
    CUDA.reclaim()
    used = CUDA.used_memory()
    allocation = z.data[]
    for i in 1:10
        x .= i
        exec()
        # the allocated memory is reused, and overwritten by every launch
        @test z.data[] === allocation
        @test Array(z) == fill(Float32(i + 1), 1024)
        @test Array(y) == fill(Float32(2(i + 1)), 1024)
    end
    # Physical pool accounting is unavailable with JULIA_CUDA_MEMORY_POOL=none.
    if used !== missing
        @test CUDA.used_memory() == used
    end

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

    # a capture that fails to begin leaves the capture in progress alone
    graph = capture() do
        @test_throws CuError capture(() -> nothing)
        a .+= 1
    end
    launch(instantiate(graph))
    @test Array(a) == fill(2, 4)
    a .= 1

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

@testset "graphs and executable graphs keep memory alive independently" begin
    x = CUDA.ones(Float32, 4)
    y = CUDA.zeros(Float32, 4)
    y .+= x
    memory = x.data[]
    graph = capture() do
        y .+= x
        CUDA.unsafe_free!(x)
        GC.gc(true)
    end
    first_exec, second_exec = instantiate(graph), instantiate(graph)
    leases = Base.@atomic memory.leases

    # memory stays leased when destroying a graph fails
    @test_throws ErrorException CUDACore.release_now(graph; destroy=_ -> error("injected graph destroy failure"))
    @test (Base.@atomic memory.leases) == leases
    finalize(graph)
    CUDA.reclaim()
    z = CUDA.fill(42f0, 4)
    first_exec(); second_exec()
    @test Array(y) == fill(3f0, 4)
    @test Array(z) == fill(42f0, 4)

    @test_throws ErrorException CUDACore.release_now(first_exec; destroy=_ -> error("injected exec destroy failure"))
    @test (Base.@atomic memory.leases) == leases - 1
    finalize(first_exec)
    CUDA.reclaim()
    z = CUDA.fill(42f0, 4)
    second_exec()
    @test Array(y) == fill(4f0, 4)
    @test Array(z) == fill(42f0, 4)
    finalize(second_exec)
    CUDA.reclaim()
end

# (not inlined, so that the object doesn't end up in a GC root of the caller)
@noinline isalive(weak::WeakRef) = weak.value !== nothing

@testset "graphs keep wrapped memory alive" begin
    # (in a function, so that no temporaries keep the wrapped array alive)
    function check(M)
        owner = fill(2f0, 1024)
        weak = WeakRef(owner)
        wrapped = unsafe_wrap(CuArray{Float32,1,M}, owner)
        out = CUDA.zeros(Float32, 1024)
        out .= wrapped
        graph = capture() do
            out .= wrapped
            CUDA.unsafe_free!(wrapped)
        end
        exec = instantiate(graph)
        owner = wrapped = nothing
        GC.gc(true)
        CUDA.reclaim()
        @test isalive(weak)
        finalize(graph)
        CUDA.reclaim()
        GC.gc(true)
        @test isalive(weak)
        out .= 0
        exec()
        @test Array(out) == fill(2f0, 1024)

        # the last lease ends with the executable graph
        finalize(exec)
        CUDA.reclaim()
        GC.gc(true)
        @test !isalive(weak)
    end
    check(CUDA.HostMemory)
    CUDACore.supports_hmm(device()) && check(CUDA.UnifiedMemory)
end

@testset "recycled stream capturing a new generation" begin
    s = CuStream()
    old = CUDA.ones(Float32, 4)
    memory = old.data[]
    lock(memory.lock) do
        CUDACore.take_ownership!(memory; stream=s)
    end
    synchronize(s)
    owner = @async nothing
    wait(owner)
    pool = [CUDACore.PooledStream(s, WeakRef(owner))]
    lock(CUDACore.stream_pool_lock) do
        @test CUDACore.claim_stream!(pool, current_task()) === s
    end
    y = CUDA.zeros(Float32, 4)
    y .+= 1
    graph = stream!(s) do
        capture() do
            @test CUDACore.pending_work(memory) === nothing
            lock(CUDACore.stream_disposal_lock) do
                handle, _ = CUDACore.release_stream(s, memory.stream_ctx, memory.generation)
                @test handle == CUDACore.disposal_stream(context())
            end
            CUDA.unsafe_free!(old)
            GC.gc(true)
            y .+= 1
        end
    end
    @test length(graph) == 1
    instantiate(graph)()
    @test Array(y) == fill(2f0, 4)
end

@testset "captured allocation provenance" begin
    allocated = Ref{CuArray{Float32,1}}()
    graph = capture() do
        allocated[] = CuArray{Float32}(undef, 16)
    end
    @test allocated[].data[].owned_allocation
    CUDA.unsafe_free!(allocated[])
    finalize(graph)
    CUDA.reclaim()
end

@testset "allocations owned by graph nodes" begin
    if CUDACore.driver_version() >= v"11.4" && CUDACore.memory_pools_supported(device())
        # Model a library allocating directly through CUDA while being captured, even
        # when CUDA.jl's own allocator is disabled.
        CUDA.reclaim()
        function graph_used()
            bytes = Ref{UInt64}()
            CUDACore.cuDeviceGetGraphMemAttribute(device(),
                CUDACore.CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT, bytes)
            bytes[]
        end
        before = graph_used()
        ptr = Ref{CUDACore.CUdeviceptr}()
        graph = capture() do
            CUDACore.cuMemAllocAsync(ptr, 4096, stream())
            CUDACore.cuMemsetD8Async(ptr[], 0x2a, 4096, stream())
        end
        exec = instantiate(graph)
        output = Vector{UInt8}(undef, 4096)
        for _ in 1:2
            exec()
            GC.@preserve output unsafe_copyto!(pointer(output),
                reinterpret(CuPtr{UInt8}, ptr[]), length(output))
            @test all(==(0x2a), output)
        end
        finalize(exec)
        finalize(graph)
        CUDA.reclaim()
        @test graph_used() <= before
    end
end
