@test has_cuda(true)
@test has_cuda_gpu(true)

# The "CUDA hasn't been initialized yet" assertions only hold in a Julia
# process that truly has a fresh runtime state, which isn't the case for us:
# ParallelTestRunner's `init_worker_code` runs `setup.jl`, which already
# touches `CUDA.functional`, `precompile_runtime`, the memory pool, etc.
# Run the fresh-state checks in a subprocess.
@testset "initialization semantics (subprocess)" begin
    script = """
        using CUDA, Test
        # the API shouldn't have been initialized
        @test_throws UndefRefError current_context()
        @test_throws UndefRefError current_device()

        ctx = context()
        dev = device()

        # querying Julia's side of things shouldn't cause initialization
        @test_throws UndefRefError current_context()
        @test_throws UndefRefError current_device()

        # now cause initialization
        a = CuArray([42])
        @test current_context() == ctx
        @test current_device() == dev
    """
    proc, _, _ = julia_exec(`-e $script`)
    @test success(proc)
end

ctx = context()
dev = device()

# ... on a different task
task = @async begin
    context()
end
@test ctx == fetch(task)

device!(CuDevice(0))
@test device!(()->true, CuDevice(0))
@inferred device!(()->42, CuDevice(0))

context!(ctx)
@test context!(()->true, ctx)
@inferred context!(()->42, ctx)

# setting flags is only possible before the primary context is active
@test_throws ErrorException device!(0, CUDA.CTX_SCHED_YIELD)

# test the device selection functionality
if length(devices()) > 1
    device!(0)
    device!(1) do
        @test device() == CuDevice(1)
    end
    @test device() == CuDevice(0)

    device!(1)
    @test device() == CuDevice(1)
end

# test that each task can work with devices independently from other tasks
if length(devices()) > 1
    device!(0)
    @test device() == CuDevice(0)

    task = @async begin
        device!(1)
        @test device() == CuDevice(1)
    end
    fetch(task)

    @test device() == CuDevice(0)

    # math_mode
    old_mm = CUDA.math_mode()
    old_prec = CUDA.math_precision()
    CUDA.math_mode!(CUDA.PEDANTIC_MATH)
    @test CUDA.math_mode() == CUDA.PEDANTIC_MATH
    CUDA.math_mode!(CUDA.PEDANTIC_MATH; precision=:Float16)
    @test CUDA.math_precision() == :Float16
    CUDA.math_mode!(old_mm; precision=old_prec)
    # ensure the values we tested here aren't the defaults
    @test CUDA.math_mode() != CUDA.PEDANTIC_MATH
    @test CUDA.math_precision() != :Float16

    # tasks on multiple threads
    Threads.@threads for d in 0:1
        for x in 1:100  # give threads a chance to trample over each other
            device!(d)
            yield()
            @test device() == CuDevice(d)
            yield()

            sleep(rand(0.001:0.001:0.01))

            device!(1-d)
            yield()
            @test device() == CuDevice(1-d)
            yield()
        end
    end
    @test device() == CuDevice(0)
end

@test deviceid(device()) >= 0
@test deviceid(CuDevice(0)) == 0
if length(devices()) > 1
    @test deviceid(CuDevice(1)) == 1
end


## default streams

default_s = stream()
s = CuStream()
@test s != default_s

# test stream switching
let
    stream!(s)
    @test stream() == s
    stream!(default_s)
    @test stream() == default_s
end
stream!(s) do
    @test stream() == s
end
@test stream() == default_s

# default stream in task
task = @async begin
    stream()
end
@test fetch(task) != default_s
@test stream() == default_s

# test stream switching in tasks
task = @async begin
    stream!(s)
    stream()
end
@test fetch(task) == s
@test stream() == default_s

function spin(cycles)
    t0 = clock(UInt64)
    while clock(UInt64) - t0 < cycles end
    return
end

@testset "stream recycling" begin
    idle_limit = CUDACore.STREAM_POOL_IDLE
    pool() = CUDACore.stream_pools[context()]
    function finished(entry)
        owner = entry.owner.value
        return owner === nothing || istaskdone(owner)
    end

    # call `f` with the streams of `n` tasks that are alive at the same time
    function with_concurrent_streams(f, n)
        ready = Channel{CuStream}(Inf)
        release = Base.Event()
        tasks = [Threads.@spawn begin
                     try
                         put!(ready, stream())
                     catch err
                         # don't leave the caller waiting for our stream
                         close(ready, err)
                         rethrow()
                     end
                     wait(release)
                 end for _ in 1:n]
        try
            f([take!(ready) for _ in tasks])
        finally
            notify(release)
            foreach(wait, tasks)
        end
    end

    # finished tasks hand their stream to new ones, without having to wait for the GC
    tasks = Task[]
    streams = map(1:2idle_limit) do _
        task = Threads.@spawn stream()
        push!(tasks, task)
        fetch(task)
    end
    @test length(unique(streams)) <= idle_limit
    @test all(s -> any(entry -> entry.stream == s, pool()), streams)

    # tasks running at the same time never share a stream, even beyond the pool's size,
    # but only a limited number of idle streams is kept around afterwards
    streams = with_concurrent_streams(identity, idle_limit+8)
    @test allunique(streams)
    foreach(synchronize, streams)
    @test fetch(Threads.@spawn stream()) in streams
    @test count(finished, pool()) <= idle_limit+1

    # tasks that keep their stream don't prevent others from being recycled
    with_concurrent_streams(idle_limit) do _
        streams = [fetch(Threads.@spawn stream()) for _ in 1:8]
        @test length(unique(streams)) <= 2
    end

    # a stream that still has work queued isn't handed to another task
    busy = fetch(Threads.@spawn begin
        @cuda spin(1_000_000_000)
        stream()
    end)
    with_concurrent_streams(idle_limit) do streams
        if !CUDA.isdone(busy)
            @test !(busy in streams)
        end
    end
    synchronize(busy)

    # streams that can't be used anymore are removed from the pool
    s = fetch(Threads.@spawn stream())
    CUDA.unsafe_destroy!(s)
    @test fetch(Threads.@spawn stream()) !== s
    @test !any(entry -> entry.stream === s, pool())

    # memory knows that the work of a stream's previous owner has finished, so it doesn't
    # wait for the new owner, nor gets released on its stream (which the new owner may be
    # capturing)
    a = fetch(Threads.@spawn begin
        a = CuArray([42])
        synchronize()
        a
    end)
    with_concurrent_streams(idle_limit) do streams
        @test a.data[].stream in streams
        @test CUDACore.recycled(a.data[])
        managed = a.data[]
        @lock CUDACore.stream_disposal_lock begin
            handle, _ = CUDACore.release_stream(managed.stream, managed.stream_ctx,
                                                managed.generation)
            @test handle == CUDACore.disposal_stream(context())
        end
        @test Array(a) == [42]
    end

    # looking for a stream to recycle doesn't break graph capture
    capture() do
        @test fetch(Threads.@spawn stream()) != stream()
    end
end

@testset "issue 1331: repeated initialization failure should stick" begin
    script = """
        using CUDA, Test
        @test !CUDA.functional()
        @test !CUDA.functional()
    """

    proc, out, err = julia_exec(`-e $script`, "CUDA_VISIBLE_DEVICES"=>"-1")
    @test success(proc)
end
