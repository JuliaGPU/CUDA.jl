@test has_cuda(true)
@test has_cuda_gpu(true)

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

# setting flags is only possible on a new context
@test_throws ErrorException device!(0, CUDA.CU_CTX_SCHED_YIELD)
if CUDA.driver_version() >= v"12"
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

    # ensure that resetting the device really does get rid of the context
    if has_nvml()
        pid = getpid()
        try
            cuda_dev = device()
            mig = uuid(cuda_dev) != parent_uuid(cuda_dev)
            nvml_dev = NVML.Device(uuid(cuda_dev); mig)
            @test haskey(NVML.compute_processes(nvml_dev), pid)
            device_reset!()
            @test !haskey(NVML.compute_processes(nvml_dev), pid)
        catch err
            isa(err, NVML.NVMLError) || rethrow()
            err.code in [NVML.ERROR_NOT_SUPPORTED, NVML.ERROR_NO_PERMISSION] || rethrow()
        end
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

    if CUDA.driver_version() >= v"12"
        # reset on a task
        task = @async begin
            device!(1)
            device_reset!()
        end
        fetch(task)
        @test device() == CuDevice(0)
    end

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

@testset "issue 1331: repeated initialization failure should stick" begin
    script = """
        using CUDA, Test
        @test !CUDA.functional()
        @test !CUDA.functional()
    """

    proc, out, err = julia_exec(`-e $script`, "CUDA_VISIBLE_DEVICES"=>"-1")
    @test success(proc)
end


## allocations

@test @allocated(current_context()) == 0
@test @allocated(context()) == 0
@test @allocated(stream()) == 0
@test @allocated(device()) == 0
