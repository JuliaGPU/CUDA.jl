# the same tests for every device memory allocator, whichever `JULIA_CUDA_MEMORY_POOL`
# selects, by constructing them directly

using CUDACore: DirectAllocator, DriverPool, CachingAllocator, try_allocate, release, used,
                reserved, trim!, purge!, generation, name

const MiB = 1 << 20

function test_allocators(; limits=(soft=UInt(0), hard=UInt(0)))
    dev = device()
    allocators = Any[DirectAllocator(dev; limits), CachingAllocator(dev; limits)]
    if CUDACore.memory_pools_supported(dev)
        push!(allocators, DriverPool(dev; limits, install=false))
    end
    return allocators
end

# allocate for use on `stream`, and free after the work submitted to it
allocate_on(a, sz, stream) = try_allocate(a, sz, (; CUDACore.active_state()..., stream))
release_on(a, mem, stream) = release(a, mem, stream, context(), generation(stream))

as_array(mem, n=sizeof(mem) ÷ 4) = unsafe_wrap(CuArray, convert(CuPtr{UInt32}, mem), n)
overlaps(x, y) = !(x.ptr + sizeof(x) <= y.ptr || y.ptr + sizeof(y) <= x.ptr)

# wait for the host to open a gate, then copy `src` to `dst`
function gated_copy(gate::Ptr{UInt32}, dst, src, cycles)
    if threadIdx().x == 1
        gate_kernel(gate, cycles)
    end
    sync_threads()
    for i in threadIdx().x:blockDim().x:length(src)
        @inbounds dst[i] = src[i]
    end
    return
end

# make work on `stream` read `mem` once the gate opens, and call `f` while it's waiting
function while_reading(f, mem, stream)
    gate = CuVector{UInt32,CUDA.HostMemory}(undef, 3)
    out = CuArray{UInt32}(undef, sizeof(mem) ÷ 4)
    GC.@preserve gate begin
        cpu = pointer(gate; type=CUDA.HostMemory)
        gpu = reinterpret(Ptr{UInt32}, pointer(gate))
        unsafe_store!(cpu, UInt32(1))
        # (compiled beforehand, as loading code waits for the GPU)
        @cuda threads=256 stream=stream gated_copy(gpu, out, as_array(mem), gate_timeout())
        synchronize(stream)
        with_watchdog() do
            for i in 1:3
                unsafe_store!(cpu, UInt32(0), i)
            end
            try
                @cuda threads=256 stream=stream gated_copy(gpu, out, as_array(mem),
                                                           gate_timeout())
                t0 = time()
                while unsafe_load(cpu, 3, :acquire) == 0 && time() - t0 < 60 end
                @test unsafe_load(cpu, 3) == 1
                f()
            finally
                unsafe_store!(cpu, UInt32(1), :release)
                synchronize(stream)
            end
        end
        @test unsafe_load(cpu, 2) == 0
    end
    return Array(out)
end

@testset "$(name(a))" for a in test_allocators()
    @testset "allocate and free" begin
        s = stream()
        used0 = used(a)
        mem = allocate_on(a, 4MiB, s)
        @test mem isa CUDA.DeviceMemory && sizeof(mem) == 4MiB
        @test used(a) >= used0 + 4MiB
        @test reserved(a) >= used(a)
        x = as_array(mem)
        fill!(x, 42)
        @test all(==(42), Array(x))
        release_on(a, mem, s)
        @test used(a) == used0
    end

    @testset "reuse on the same stream" begin
        s = CuStream()
        x = allocate_on(a, MiB, s)
        stream!(s) do
            fill!(as_array(x), 1)
        end
        out = while_reading(x, s) do
            release_on(a, x, s)
            # memory freed on a stream can be reused there right away, as long as later
            # work on it doesn't overwrite what earlier work still reads
            y = allocate_on(a, MiB, s)
            stream!(s) do
                fill!(as_array(y), 2)
            end
            release_on(a, y, s)
        end
        @test all(==(1), out)
    end

    @testset "reuse on another stream" begin
        s1, s2 = CuStream(), CuStream()
        # make the memory that is about to be freed the only memory that could be reused
        # (a cached segment has room for two of these)
        synchronize()
        purge!(a)
        x = allocate_on(a, MiB, s1)
        filler = allocate_on(a, MiB, s1)
        stream!(s1) do
            fill!(as_array(x), 1)
        end
        out = while_reading(x, s1) do
            release_on(a, x, s1)
            y = allocate_on(a, MiB, s2)
            stream!(s2) do
                fill!(as_array(y), 2)
            end
            if overlaps(x, y)
                # only allowed if the work on the other stream waits for the earlier use
                sleep(0.1)
                @test !CUDACore.isdone(s2)
            end
            release_on(a, y, s2)
        end
        @test all(==(1), out)
        release_on(a, filler, s1)
    end

    @testset "statistics and reclaiming" begin
        CUDA.reclaim()
        purge!(a)
        used0, reserved0 = used(a), reserved(a)
        s = stream()
        mems = [allocate_on(a, sz, s) for sz in (512, 3MiB, 12MiB, 64MiB)]
        @test used(a) >= used0 + 79MiB
        @test reserved(a) >= used(a)
        foreach(mem -> release_on(a, mem, s), mems)
        @test used(a) == used0
        # once the GPU is done with it, reclaiming memory returns it all
        CUDA.reclaim()
        purge!(a)
        @test reserved(a) <= reserved0
    end

    @testset "freed after its stream was destroyed" begin
        s = CuStream()
        mem = allocate_on(a, MiB, s)
        stream!(s) do
            fill!(as_array(mem), 1)
        end
        gen = generation(s)
        finalize(s)
        CUDACore.drain_retired()
        @test !CUDACore.isvalid(s)
        used0 = used(a)
        release(a, mem, s, context(), gen)
        @test used(a) == used0 - MiB
        CUDA.reclaim()
    end

    @testset "context destroyed while freed memory is returned" begin
        used0, reserved0 = used(a), reserved(a)
        CuContext(device()) do ctx
            context!(ctx) do
                s = CuStream()
                mem = allocate_on(a, MiB, s)
                stream!(s) do
                    fill!(as_array(mem), 1)
                end
                release(a, mem, s, ctx, generation(s))
                CUDACore.record_free_events()
            end
        end
        CUDA.reclaim()
        purge!(a)
        @test used(a) == used0
        @test reserved(a) <= reserved0
    end

    @testset "freed after its context was destroyed" begin
        used0 = used(a)
        local mem, s, ctx
        CuContext(device()) do c
            ctx = c
            context!(ctx) do
                s = CuStream()
                mem = allocate_on(a, MiB, s)
                stream!(s) do
                    fill!(as_array(mem), 1)
                end
            end
        end
        @test !CUDACore.isvalid(ctx)
        release(a, mem, s, ctx, generation(s))
        @test used(a) == used0
        CUDA.reclaim()
    end

    @testset "graph capture" begin
        # allocate on the allocation stream while capturing, like `capture_alloc` does
        local mem
        graph = capture() do
            side = CUDACore.allocation_stream(context())
            mem = CUDACore.relaxed_capture_mode() do
                allocate_on(a, MiB, side)
            end
            fill!(as_array(mem), 3)
        end
        synchronize(CUDACore.allocation_stream(context()))
        exec = instantiate(graph)
        exec()
        synchronize()
        @test all(==(3), Array(as_array(mem)))
        release_on(a, mem, stream())
    end
end

@testset "hard limit: $(name(a))" for a in test_allocators(; limits=(soft=UInt(0), hard=UInt(64MiB)))
    s = stream()
    x = allocate_on(a, 48MiB, s)
    @test x !== nothing
    @test allocate_on(a, 48MiB, s) === nothing
    release_on(a, x, s)
    # once freed (and returned to the driver), the memory can be allocated again
    CUDA.reclaim()
    purge!(a)
    y = allocate_on(a, 48MiB, s)
    @test y !== nothing
    release_on(a, y, s)
end
