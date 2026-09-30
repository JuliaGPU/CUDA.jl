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

function priority_increment!(a, cycles)
    spin(cycles)
    a[1] += 1
    return
end

@testset "stream recycling" begin
    idle_limit = CUDACore.STREAM_POOL_IDLE

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
    streams = [fetch(Threads.@spawn begin
                   s = stream()
                   synchronize(s)
                   s
               end) for _ in 1:1024]
    @test length(unique(streams)) <= idle_limit + 1

    # tasks running at the same time never share a stream, even beyond the number of idle
    # streams that are kept around
    streams = with_concurrent_streams(identity, idle_limit+8)
    @test allunique(streams)
    foreach(synchronize, streams)
    @test fetch(Threads.@spawn stream()) in streams
    # (peeks at the pool, as there's no way to observe which idle streams it has dropped)
    finished(entry) = (owner = entry.owner.value; owner === nothing || istaskdone(owner))
    pool = CUDACore.stream_pools[(context(), Cint(0), CUDACore.STREAM_DEFAULT)]
    @test count(finished, pool) <= idle_limit + 1

    # tasks that keep their stream don't prevent others from being recycled
    with_concurrent_streams(idle_limit) do _
        streams = [fetch(Threads.@spawn stream()) for _ in 1:8]
        @test length(unique(streams)) <= 2
    end

    # a stream that still has work queued isn't handed to another task
    gate = UInt32[1, 0, 0]  # (is open, timed out, started), see `gate_kernel`
    gpu_gate = unsafe_wrap(CuArray, gate)
    gate_ptr = reinterpret(Ptr{UInt32}, pointer(gpu_gate))
    # (compiled beforehand, as loading code waits for the GPU)
    @cuda gate_kernel(gate_ptr, gate_timeout())
    synchronize()
    GC.@preserve gpu_gate begin
        gate[1] = 0
        busy = fetch(Threads.@spawn begin
            @cuda gate_kernel(gate_ptr, gate_timeout())
            stream()
        end)
        try
            with_concurrent_streams(idle_limit) do streams
                @test !(busy in streams)
            end
        finally
            unsafe_store!(pointer(gate), UInt32(1), :release)
            synchronize(busy)
        end
        @test gate[2] == 0
    end

    # streams that have been destroyed aren't handed out again
    s = fetch(Threads.@spawn stream())
    CUDA.unsafe_destroy!(s)
    @test fetch(Threads.@spawn stream()) !== s

    # looking for a stream to recycle doesn't break graph capture (by a task that isn't
    # spawned during the capture, and thus doesn't take part in it)
    go = Base.Event()
    other = Threads.@spawn (wait(go); stream())
    capture() do
        notify(go)
        @test fetch(other) != stream()
    end
end

@testset "task priority" begin
    priority! = CUDA.priority!
    high = last(priority_range())
    @test_throws ArgumentError priority!(:invalid)
    @test_throws ArgumentError priority!(high - 1)
    @test_throws ArgumentError priority!(false)

    # a priority can be chosen before the task first uses the GPU
    fetch(Threads.@spawn begin
        priority!(:high)
        @test priority() == high
        high_stream = stream()
        @test priority(high_stream) == high

        priority!(:high)
        @test stream() === high_stream
        priority!(high)
        @test stream() === high_stream
        priority!(:normal)
        normal_stream = stream()
        @test priority() == 0
        priority!(:low)
        @test priority() == first(priority_range())
        priority!(:high)
        @test stream() === high_stream
        @test high == 0 || normal_stream !== high_stream
    end)

    fetch(Threads.@spawn begin
        high_stream = priority!(:high) do
            s = stream()
            priority!(:normal) do
                @test priority() == 0
            end
            @test stream() === s
            s
        end
        @test priority() == 0
        @test high == 0 || stream() !== high_stream
    end)

    normal_stream = stream()
    selected = priority!(:high) do
        @test priority() == high
        stream()
    end
    @test stream() === normal_stream
    @test priority() == 0
    @test priority(selected) == high
    @test_throws ErrorException priority!(:high) do
        error("scope failed")
    end
    @test stream() === normal_stream

    explicit = CuStream(; flags=CUDACore.STREAM_NON_BLOCKING)
    stream!(explicit) do
        priority!(:high) do
            @test priority() == high
            @test high == 0 || stream() !== explicit
            @test CUDACore.stream_flags(stream()) == CUDACore.STREAM_NON_BLOCKING
        end
        @test stream() === explicit
    end
    @test stream() === normal_stream

    if high != 0
        fetch(Threads.@spawn begin
            priority!(:high)
            stream!(explicit) do
                priority!(:normal)
            end
            @test priority() == high
            @test priority(stream()) == high
        end)
    end

    # KernelInterface uses the same task-local selection, without new streams on repeats
    KI = CUDACore.CUDAKernels.KI
    fetch(Threads.@spawn begin
        KI.priority!(CUDACore.CUDABackend(), :high)
        s = stream()
        KI.priority!(CUDACore.CUDABackend(), :high)
        @test stream() === s
    end)

    # changing priority during capture cannot move the task away from the capturing stream
    capture() do
        s = stream()
        if high == 0
            priority!(:high)
        else
            @test_throws ArgumentError priority!(:high)
        end
        @test stream() === s
        priority!(:normal)
        @test_throws ArgumentError stream!(() -> nothing, explicit)
    end

    if high != 0
        fetch(Threads.@spawn begin
            first_stream = stream()
            @cuda spin(200_000_000)
            priority!(:high)
            second_stream = stream()
            synchronize()
            @test CUDA.isdone(first_stream)
            @cuda spin(200_000_000)
            priority!(:normal)
            synchronize()
            @test CUDA.isdone(second_stream)
        end)
    end

    a = CuArray(Int32[0])
    @cuda priority_increment!(a, 0) # compile before switching priorities
    synchronize()
    fetch(Threads.@spawn begin
        # Work on an explicit stream is not ordered by the task's priority handoff.
        # Using the array on the new stream must synchronize with this kernel.
        @cuda stream=explicit priority_increment!(a, 200_000_000)
        priority!(:high)
        @cuda priority_increment!(a, 0)
        priority!(:normal)
        @cuda priority_increment!(a, 0)
        synchronize()
    end)
    @test Array(a) == Int32[4]
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
