@test has_cuda(true)
@test has_cuda_gpu(true)

# the API shouldn't have been initialized
@test CuCurrentContext() == nothing
@not_if_memcheck @test CuCurrentDevice() == nothing

task_cb = Any[nothing for tid in 1:Threads.nthreads()]
CUDA.attaskswitch() do
    task_cb[Threads.threadid()] = current_task()
end

device_switch_cb = Any[nothing for tid in 1:Threads.nthreads()]
CUDA.atdeviceswitch() do
    device_switch_cb[Threads.threadid()] = (dev=device(), ctx=context())
end

device_reset_cb = Any[nothing for tid in 1:Threads.nthreads()]
CUDA.atdevicereset() do dev
    device_reset_cb[Threads.threadid()] = dev
end

function reset_cb()
    fill!(task_cb, nothing)
    fill!(device_switch_cb, nothing)
    fill!(device_reset_cb, nothing)
end

# now cause initialization
ctx = context()
dev = device()
@test CuCurrentContext() == ctx
@test CuCurrentDevice() == dev
@test task_cb[1] == current_task()
@test device_switch_cb[1].ctx == ctx
@test device_switch_cb[1].dev == dev

reset_cb()

# ... on a different task
task = @async begin
    context()
end
@test ctx == fetch(task)
@test task_cb[1] == task
@test device_switch_cb[1].ctx == ctx
@test device_switch_cb[1].dev == dev

reset_cb()

# ... back to the main task
ctx = context()
dev = device()
@test task_cb[1] == current_task()
@test device_switch_cb[1] == nothing

device!(CuDevice(0))
device!(CuDevice(0)) do
    nothing
end

context!(ctx)
context!(ctx) do
    nothing
end

@test_throws ErrorException device!(0, CUDA.CU_CTX_SCHED_YIELD)

reset_cb()

device_reset!()

@test device_reset_cb[1] == CuDevice(0)

reset_cb()

device!(0, CUDA.CU_CTX_SCHED_YIELD)
@test task_cb[1] == nothing
@test device_switch_cb[1].dev == CuDevice(0)

# reset on a different task
let ctx = context()
    @test CUDA.isvalid(ctx)
    @test ctx == fetch(@async context())

    @sync @async device_reset!()

    @test CUDA.isvalid(context())
    @test ctx != context()
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
