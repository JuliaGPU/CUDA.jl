CUDACore.pool_alloc(0)

@test_throws OutOfGPUMemoryError CuArray{Int}(undef, 10^20)

try
    CuArray{Int}(undef, 10^20)
catch e
    @test startswith(sprint(showerror, e), "Out of GPU memory")
end


@testset "@allocated" begin
    @test (CUDA.@allocated CuArray{Int32}(undef,1)) == 4
end

@testset "@timed" begin
    out = CUDA.@timed CuArray{Int32}(undef, 1)
    @test isa(out.value, CuArray{Int32})
    @test out.gpu_bytes > 0
end

@testset "@time" begin
    ret, out = @grab_output CUDA.@time CuArray{Int32}(undef, 1)
    @test isa(ret, CuArray{Int32})
    @test occursin("1 GPU allocation: 4 bytes", out)

    x = CuArray{Int32}(undef, 6)
    ret, out = @grab_output CUDA.@time Base.unsafe_wrap(CuArray, pointer(x), (2, 3))
    @test isa(ret, CuArray{Int32})
    @test !occursin("GPU allocation", out)
end

@testset "reclaim" begin
    # every level should run without erroring, on any allocator
    for level in instances(CUDACore.ReclaimLevel)
        CUDA.reclaim(level)
    end
    CUDA.reclaim()

    # `retry_reclaim` returns the block's result, retrying while `retry_if` holds
    @test CUDACore.retry_reclaim(isequal(42)) do
            42
        end == 42
    @test CUDACore.retry_reclaim(isequal(42)) do
            41
        end == 41

    # `retry_reclaim` escalates: count how many calls it takes to stop retrying
    let n = Ref(0)
        ret = CUDACore.retry_reclaim(ret -> ret < 2) do
            n[] += 1
            n[]
        end
        @test ret == 2
        @test n[] == 2
    end
end

@testset "pool_status" begin
    status = sprint(CUDA.pool_status)
    @test occursin("Memory allocator: $(CUDACore.name(CUDACore.allocator(device())))", status)
    @test CUDA.used_memory() isa Int
    @test CUDA.cached_memory() >= CUDA.used_memory()
end

@testset "allocation origin" begin
    mem = CUDACore.alloc(CUDA.DeviceMemory, 1024)
    @test mem.origin == CUDACore.DIRECT_MEMORY && !mem.async
    CUDACore.free(mem)
    if CUDACore.memory_pools_supported(device())
        mem = CUDACore.alloc(CUDA.DeviceMemory, 1024; async=true)
        @test mem.origin == CUDACore.POOL_MEMORY && mem.async
        CUDACore.free(mem)
    end
end

@testset "host memory pools don't depend on the device memory allocator" begin
    # (also tested with `JULIA_CUDA_MEMORY_POOL=none` in CI)
    if CUDACore.host_memory_pools && CUDACore.driver_version() >= v"13.0" &&
       all(dev -> attribute(dev, CUDA.DEVICE_ATTRIBUTE_HOST_MEMORY_POOLS_SUPPORTED) == 1,
           devices())
        @test CUDACore.host_pool() !== nothing
    end
end

@testset "live bytes return to zero" begin
    # (the hard limit is checked against them when not using a pool)
    live = CUDACore.memory_stats().live
    for _ in 1:10
        CUDA.unsafe_free!(CuArray{UInt8}(undef, 1024))
    end
    @test CUDACore.memory_stats().live == live
end

@testset "parse_limit" begin
    @test CUDACore.parse_limit("8000kB") == UInt(8000*1000)
    @test CUDACore.parse_limit("8000MB") == UInt(8000*1000*1000)
    @test CUDACore.parse_limit("8GB")    == UInt(8*1000*1000*1000)

    @test CUDACore.parse_limit("8KiB")   == UInt(8*1024)
    @test CUDACore.parse_limit("8MiB")   == UInt(8*1024*1024)
    @test CUDACore.parse_limit("8GiB")   == UInt(8*1024*1024*1024)
end

@testset "one reclaim resolves PTDS releases" begin
    CUDA.reclaim()
    for M in (CUDA.DeviceMemory, CUDA.HostMemory, CUDA.UnifiedMemory)
        CUDA.stream!(CUDA.per_thread_stream()) do
            a = CuArray{UInt8,1,M}(undef, 513)
            CUDA.unsafe_free!(a)
        end
        CUDA.reclaim()
        @test (Base.@atomic CUDACore.host_cache.bytes) == 0
        @test (Base.@atomic CUDACore.unified_cache.bytes) == 0
    end
end

# memory that isn't allocated from a pool is freed in stream order, and is only returned to
# the driver after synchronizing. NVML reports how much this process uses, unlike
# `CUDA.free_memory()`, which other processes affect.
const unpooled_size = 512 * 2^20
function free_unpooled()
    mem = CUDACore.alloc(CUDACore.DeviceMemory, unpooled_size)
    a = unsafe_wrap(CuArray, convert(CuPtr{UInt8}, mem), unpooled_size; own=true)
    a .= 0x01
    CUDA.unsafe_free!(a)
end

@testset "reclaiming returns memory that isn't pooled" begin
    CUDA.reclaim()
    before = gpu_rss_nvml()
    if before !== missing
        free_unpooled()
        CUDA.reclaim()
        @test gpu_rss_nvml() < before + unpooled_size ÷ 2
    end
end

@testset "memory that isn't pooled is returned without reclaiming" begin
    # (without stream-ordered frees, it is only freed when reclaiming memory)
    before = gpu_rss_nvml()
    if before !== missing && CUDACore.async_free_supported(device())
        free_unpooled()
        t0 = time()
        while gpu_rss_nvml() >= before + unpooled_size ÷ 2 && time() - t0 < 10
            sleep(0.1)
        end
        @test gpu_rss_nvml() < before + unpooled_size ÷ 2
    end
end

@testset "early_gc!" begin
    # with a 64 MiB limit, keeping a 56 MiB array alive puts the pool above the pressure
    # at which allocations run the GC
    script = """
        using CUDA, Test

        function alloc_collects()
            # a collection resets Julia's allocation counter, so that the few bytes the
            # loop allocates can't make Julia collect by itself
            GC.gc()
            before = Base.gc_num().pause
            for _ in 1:4
                CuArray{UInt8}(undef, 1024)
            end
            Base.gc_num().pause > before
        end
        disabled() = CUDA.early_gc!(alloc_collects, false)
        disabled_task() = CUDA.early_gc!(false) do
            fetch(Threads.@spawn alloc_collects())
        end
        reenabled() = CUDA.early_gc!(false) do
            CUDA.early_gc!(alloc_collects, true)
        end

        # compile everything before there's pressure, so that the compiler doesn't run the GC
        @assert !disabled() && !disabled_task() && !reenabled()

        keep = CuArray{UInt8}(undef, 56*2^20)
        @test !disabled()
        @test !disabled_task()
        @test reenabled()
        """
    proc, out, err = julia_exec(`-e $script`, "JULIA_CUDA_HARD_MEMORY_LIMIT" => "64MiB")
    if !success(proc)
        println(out)
        println(err)
    end
    @test success(proc)
end
