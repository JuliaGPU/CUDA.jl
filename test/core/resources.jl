using GPUArrays
using CUDACore: resource_finalizer

@testset "reclaim frees memory that just became unreachable" begin
    # (in a function, so that the array isn't kept alive by the test scope)
    @noinline allocate() = (CuArray{UInt8}(undef, 256 * 2^20) .= 1; nothing)
    CUDA.reclaim()
    before = CUDA.used_memory()
    allocate()
    CUDA.reclaim()
    if before !== missing
        @test CUDA.used_memory() <= before
    end
end

@testset "resource_finalizer" begin
    calls = Tuple{Symbol,Bool}[]
    @noinline function register!(calls)
        resource_finalizer(Ref(0); blocking=false, ctx=nothing) do _
            push!(calls, (:prompt, GC.in_finalizer()))
        end
        resource_finalizer(Ref(0); ctx=nothing) do _
            push!(calls, (:blocking, GC.in_finalizer()))
        end
        return
    end
    register!(calls)
    GC.gc(true)
    # the garbage collector only queues the cleanup
    @test isempty(calls)
    # which happens on a regular task, when releasing memory...
    CUDA.pool_status(devnull)
    @test calls == [(:prompt, false)]
    # ... or when reclaiming it, for cleanup that may wait for the GPU
    CUDA.reclaim()
    @test calls == [(:prompt, false), (:blocking, false)]

    # invalid arguments are rejected when registering, not when cleaning up
    @test_throws TypeError resource_finalizer(identity, Ref(0); blocking=:invalid)
    @test_throws TypeError resource_finalizer(identity, Ref(0); ctx=:invalid)

    # failing cleanup is logged, and not retried
    calls = Int[]
    obj = Ref(1)
    resource_finalizer(obj; blocking=false, ctx=nothing) do obj
        push!(calls, obj[])
        error("injected failure")
    end
    finalize(obj)
    @test_logs (:error, r"Failed to release a GPU resource") CUDA.pool_status(devnull)
    CUDA.reclaim()
    @test calls == [1]
end

if attribute(device(), CUDA.DEVICE_ATTRIBUTE_COMPUTE_MODE) != CUDA.COMPUTEMODE_EXCLUSIVE_PROCESS
    @testset "cleanup in the registered context" begin
        original = context()
        foreign = CuContext(device())
        activate(original)
        seen = Ref{Union{Nothing,CuContext}}(nothing)
        obj = Ref(0)
        resource_finalizer(obj; ctx=foreign) do _
            seen[] = context()
        end
        finalize(obj)
        CUDA.reclaim()
        @test seen[] == foreign
        @test context() == original
        CUDA.unsafe_destroy!(foreign)
    end
end

@testset "arrays" begin
    # finalizing an array only releases its reference to the memory
    a = CuArray([1, 2, 3, 4])
    alias = reshape(a, 2, 2)
    finalize(a)
    CUDA.reclaim()
    @test Array(alias) == [1 3; 2 4]
    data = alias.data
    finalize(alias)
    CUDA.reclaim()
    @test data.freed

    # also of memory that replaced the original one
    a = CuArray([1, 2])
    resize!(a, 16)
    data = a.data
    finalize(a)
    CUDA.reclaim()
    @test data.freed

    # freeing an array explicitly happens right away, and only once
    a = CuArray{UInt8}(undef, 4096)
    data = a.data
    before = Base.@atomic CUDACore.alloc_stats.free_count
    CUDA.unsafe_free!(a)
    @test data.freed
    finalize(a)
    CUDA.reclaim()
    @test (Base.@atomic CUDACore.alloc_stats.free_count) == before + 1
end

@testset "arrays from an allocation cache" begin
    cache = GPUArrays.AllocCache()
    a = GPUArrays.@cached cache CuArray([1, 2, 3, 4])
    data = a.data
    alias = reshape(a, 2, 2)
    finalize(a)
    b = GPUArrays.@cached cache CuArray([5, 6, 7, 8])
    @test b.data === data
    CUDA.reclaim()
    @test !data.freed
    @test Array(b) == [5, 6, 7, 8]
    # the cache may be invalidated before an array it allocated is released, which should
    # not release the cache's reference
    data.cached = false
    finalize(b)
    CUDA.reclaim()
    @test !data.freed
    data.cached = true
    GPUArrays.unsafe_free!(cache)
    @test data.freed
    @test Array(alias) == [5 7; 6 8]
    CUDA.unsafe_free!(alias)

    # memory that replaced the cached one is owned by the array
    a = GPUArrays.@cached cache CuArray{Int}(undef, 4)
    cached = a.data
    resize!(a, 16)
    replacement = a.data
    @test replacement !== cached
    finalize(a)
    CUDA.reclaim()
    @test replacement.freed
    @test !cached.freed
    GPUArrays.unsafe_free!(cache)
    @test cached.freed
end

@testset "handle caches" begin
    # handles that are returned from a finalizer become reusable, and are only destroyed
    # when reclaiming memory
    destroyed = Int[]
    cache = CUDACore.HandleCache{Int,Int}(identity, (key, handle) -> push!(destroyed, handle))
    CUDACore.register_reclaimable!(cache)
    @noinline function borrow(cache)
        wrapper = Ref(pop!(cache, 1))
        finalizer(x -> push!(cache, 1, x[]), wrapper)
        return WeakRef(wrapper)
    end
    weak = borrow(cache)

    # returning it doesn't need the cache's lock, which another task may hold
    ready, unlock_cache = Channel{Nothing}(1), Channel{Nothing}(1)
    holder = @async lock(cache.lock) do
        put!(ready, nothing)
        take!(unlock_cache)
    end
    take!(ready)
    try
        GC.gc(true)
        @test weak.value === nothing
    finally
        put!(unlock_cache, nothing)
        wait(holder)
    end
    CUDA.pool_status(devnull)
    @test CUDACore.handle_cache_counts(cache) == (active=0, idle=1)
    @test isempty(destroyed)
    CUDA.reclaim()
    @test destroyed == [1]
    @test CUDACore.handle_cache_counts(cache) == (active=0, idle=0)

    # resources that may depend on a handle are released before handles are destroyed,
    # and resources released by destroying a handle are released by the same reclaim
    order = Symbol[]
    cache = CUDACore.HandleCache{Int,Int}(identity, (k, v) -> begin
        push!(order, :handle)
        CUDACore.destroy_later(_ -> push!(order, :nested), nothing)
    end)
    CUDACore.register_reclaimable!(cache)
    push!(cache, 1, pop!(cache, 1))
    CUDACore.destroy_later(_ -> push!(order, :helper), nothing)
    CUDA.reclaim()
    @test order == [:helper, :handle, :nested]

    # handles whose destruction fails are not destroyed again
    calls = Int[]
    cache = CUDACore.HandleCache{Int,Int}(identity, (k, v) -> begin
        push!(calls, v)
        v == 2 && error("injected failure")
    end)
    for i in 1:3
        push!(cache, i, pop!(cache, i))
    end
    CUDA.pool_status(devnull)
    @test_logs (:error, r"Failed to release a GPU resource") empty!(cache)
    @test sort(calls) == [1, 2, 3]
    CUDA.reclaim()
    @test length(calls) == 3
end

@testset "bounded draining" begin
    # finalizers may retire many resources at once, but allocating only releases a bounded
    # number of them, and only those that were retired before
    CUDA.pool_status(devnull)
    count = Ref(0)
    for _ in 1:600
        CUDACore.retire!(CUDACore.ReleaseAction(_ -> (count[] += 1), nothing, nothing, false))
    end
    @test CUDACore.drain_retired(CUDACore.ALLOC_DRAIN_LIMIT) == 256
    @test count[] == 256
    @test CUDACore.drain_retired() == 344
    @test count[] == 600

    calls = Int[]
    CUDACore.retire!(CUDACore.ReleaseAction(nothing, nothing, false) do _
        push!(calls, 1)
        CUDACore.retire!(CUDACore.ReleaseAction(_ -> push!(calls, 2), nothing, nothing, false))
        @test CUDACore.drain_retired() == 0
    end)
    @test CUDACore.drain_retired() == 1
    @test calls == [1]
    CUDACore.drain_retired()
    @test calls == [1, 2]
end

@testset "graph capture" begin
    # nothing is released during a capture
    calls = Symbol[]
    obj = Ref(0)
    resource_finalizer(_ -> push!(calls, :released), obj; blocking=false, ctx=nothing)
    a = CuArray([1])
    b = CuArray([2])
    a .+= 1
    graph = capture() do
        a .+= 1
        finalize(obj)
        CUDA.unsafe_free!(b)
        CUDACore.drain_retired()
    end
    @test isempty(calls)
    CUDA.pool_status(devnull)
    @test calls == [:released]
    CUDA.launch(CUDA.instantiate(graph))
    @test Array(a) == [3]

    # a capture waits for resources that are being released...
    started, finish = Channel{Nothing}(1), Channel{Nothing}(1)
    releaser = @async CUDACore.releasing() do
        put!(started, nothing)
        take!(finish)
    end
    take!(started)
    capturer = @async capture(() -> nothing)
    for _ in 1:100
        yield()
    end
    @test !istaskdone(capturer)
    put!(finish, nothing)
    wait(releaser)
    @test fetch(capturer) isa CuGraph

    # ... unless releasing them may wait for the GPU
    CUDACore.releasing(; blocking=true) do
        @test_throws ErrorException capture(() -> nothing)
    end
end

@testset "stream construction from finalizers is rejected" begin
    # (a task's stream is created on first use, which can't be arranged to happen in a
    #  finalizer, so call the internal function that does)
    state = CUDACore.task_local_state!()
    result = Ref{Any}(nothing)
    @noinline function stream_finalizer(state, result)
        obj = Ref(0)
        finalizer(obj) do _
            result[] = try
                CUDACore.create_stream(state)
            catch err
                err
            end
        end
        WeakRef(obj)
    end
    weak = stream_finalizer(state, result)
    GC.gc(true)
    @test weak.value === nothing
    @test result[] isa ErrorException
    @test occursin("Cannot create a CUDA stream from a finalizer", sprint(showerror, result[]))
end
