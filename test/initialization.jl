@test has_cuda(true)
@test has_cuda_gpu(true)

# the API shouldn't have been initialized
@test CuCurrentContext() == nothing
@test CuCurrentDevice() == nothing

ctx = context()
dev = device()

# querying Julia's side of things shouldn't cause initialization
@test CuCurrentContext() == nothing
@test CuCurrentDevice() == nothing

# now cause initialization
a = CuArray([42])
@test CuCurrentContext() == ctx
@test CuCurrentDevice() == dev

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

@test_throws ErrorException device!(0, CUDA.CU_CTX_SCHED_YIELD)

if CUDA.can_reset_device()
    # NVIDIA bug #3240770
    device_reset!()

    device!(0, CUDA.CU_CTX_SCHED_YIELD)

    # reset on a different task
    let ctx = context()
        @test CUDA.isvalid(ctx)
        @test ctx == fetch(@async context())

        @sync @async device_reset!()

        @test CUDA.isvalid(context())
        @test ctx != context()
    end
end

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

    # reset on a task
    task = @async begin
        device!(1)
        device_reset!()
    end
    fetch(task)
    @test device() == CuDevice(0)

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

@test deviceid() >= 0
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
