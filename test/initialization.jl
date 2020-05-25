@test has_cuda(true)
@test has_cuda_gpu(true)

# the API shouldn't have been initialized
@test CuCurrentContext() == nothing

context_cb = Union{Nothing, CuContext}[nothing for tid in 1:Threads.nthreads()]
CUDA.atcontextswitch() do tid, ctx
    context_cb[tid] = ctx
end

task_cb = Union{Nothing, Task}[nothing for tid in 1:Threads.nthreads()]
CUDA.attaskswitch() do tid, task
    task_cb[tid] = task
end

# now cause initialization
ctx = context()
@test CuCurrentContext() == ctx
@test device() == CuDevice(0)
@test context_cb[1] == ctx
@test task_cb[1] == current_task()

fill!(context_cb, nothing)
fill!(task_cb, nothing)

# ... on a different task
task = @async begin
    context()
end
@test ctx == fetch(task)
@test context_cb[1] == nothing
@test task_cb[1] == task

device!(CuDevice(0))
device!(CuDevice(0)) do
    nothing
end

context!(ctx)
context!(ctx) do
    nothing
end

@test_throws AssertionError device!(0, CUDA.CU_CTX_SCHED_YIELD)

device_reset!()

device!(0, CUDA.CU_CTX_SCHED_YIELD)

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
