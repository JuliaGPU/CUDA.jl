# the caching allocator in situations that are hard to get right, forcing the dangerous
# interleavings with a kernel that waits for the host (see test/core/allocators.jl for the
# tests that every allocator passes)

using CUDACore: CachingAllocator, try_allocate, release, used, reserved, trim!, purge!,
                generation, inject_fault!, set_retention!

const MiB = 1 << 20

allocate_on(a, sz, stream) = try_allocate(a, sz, (; CUDACore.active_state()..., stream))
release_on(a, mem, stream) = release(a, mem, stream, context(), generation(stream))
as_array(mem, n=sizeof(mem) ÷ 4) = unsafe_wrap(CuArray, convert(CuPtr{UInt32}, mem), n)
overlaps(x, y) = !(x.ptr + sizeof(x) <= y.ptr || y.ptr + sizeof(y) <= x.ptr)

# a gate that a kernel waits for (see `gate_kernel`). its pointers are taken once, as taking
# them hands the memory to the current stream, or synchronizes with that stream.
struct Gate
    host::CuVector{UInt32,CUDA.HostMemory}
    cpu::Ptr{UInt32}
    gpu::Ptr{UInt32}
end
function Gate()
    host = CuVector{UInt32,CUDA.HostMemory}(undef, 3)
    gate = Gate(host, pointer(host; type=CUDA.HostMemory),
                reinterpret(Ptr{UInt32}, pointer(host)))
    foreach(i -> unsafe_store!(gate.cpu, UInt32(1), i), 1:3)
    # (compiled beforehand, as loading code waits for the GPU)
    @cuda gate_kernel(gate.gpu, gate_timeout())
    copyto!(CUDA.zeros(UInt32, 1), CUDA.zeros(UInt32, 1))
    synchronize()
    foreach(i -> unsafe_store!(gate.cpu, UInt32(0), i), 1:3)
    return gate
end
function wait_started(g::Gate)
    t0 = time()
    while unsafe_load(g.cpu, 3, :acquire) == 0 && time() - t0 < 60 end
    @test unsafe_load(g.cpu, 3) == 1
end
open!(g::Gate) = unsafe_store!(g.cpu, UInt32(1), :release)
timed_out(g::Gate) = unsafe_load(g.cpu, 2) == 1

# memory that is reallocated while work that reads it is still waiting for the gate must
# not overlap it, unless the work on the new memory waits for that
function check_reallocation(x, y, s_new)
    if overlaps(x, y)
        sleep(0.1)
        @test !CUDACore.isdone(s_new)
    end
end

@testset "stream activated by another task" begin
    a = CachingAllocator(device())
    gate = Gate()
    out = CuArray{UInt32}(undef, MiB ÷ 4)
    # a task's stream, activated by another task after it has finished
    s = fetch(Threads.@spawn stream())
    synchronize(s)
    freed = Channel{Any}(1)
    done = Base.Event()
    user = Threads.@spawn begin
        stream!(s)
        x = allocate_on(a, MiB, s)
        fill!(as_array(x), 1)
        filler = allocate_on(a, MiB, s)
        # (read the memory once the gate opens)
        @cuda gate_kernel(gate.gpu, gate_timeout())
        copyto!(out, as_array(x))
        release_on(a, x, s)
        put!(freed, (x, filler))
        wait(done)
        synchronize(s)
        release_on(a, filler, s)
    end
    with_watchdog() do
        try
            x, filler = take!(freed)
            wait_started(gate)
            # new tasks don't get that stream, as the other task may still use it, so the
            # memory freed on it isn't considered idle either
            for _ in 1:4
                y, s_new = fetch(Threads.@spawn begin
                    s_new = stream()
                    y = allocate_on(a, MiB, s_new)
                    fill!(as_array(y), 2)
                    y, s_new
                end)
                @test s_new != s
                check_reallocation(x, y, s_new)
                release_on(a, y, s_new)
            end
        finally
            open!(gate)
            notify(done)
            wait(user)
        end
    end
    @test !timed_out(gate)
    @test all(==(1), Array(out))
end

@testset "capture while trimming" begin
    a = CachingAllocator(device())
    set_retention!(a, 0)
    s = stream()
    x = allocate_on(a, 4MiB, s)
    release_on(a, x, s)
    synchronize()
    held = reserved(a)
    @test held >= 4MiB

    # a trim that runs during a capture doesn't release (or lose) anything
    y = CUDA.zeros(Float32, 16)
    graph = capture() do
        fetch(Threads.@spawn trim!(a))
        @test reserved(a) == held
        y .+= 1
    end
    @test reserved(a) == held
    exec = instantiate(graph)
    exec()
    @test Array(y) == ones(Float32, 16)

    # afterwards, it does
    trim!(a)
    purge!(a)
    @test reserved(a) == 0

    # and the segment can't be reused after it has been released
    z = allocate_on(a, 4MiB, s)
    @test z !== nothing && reserved(a) >= 4MiB
    release_on(a, z, s)
    purge!(a)
end

@testset "injected failures" begin
    a = CachingAllocator(device())
    s = stream()

    # failing to allocate a segment returns its credit, and is retried
    inject_fault!(a, :alloc)
    x = allocate_on(a, 3MiB, s)
    @test x !== nothing
    @test reserved(a) == 20MiB
    release_on(a, x, s)
    synchronize()
    purge!(a)
    @test reserved(a) == 0

    # a segment that couldn't be added to the cache keeps its credit until it's freed
    inject_fault!(a, :metadata)
    @test_throws ErrorException allocate_on(a, 3MiB, s)
    @test reserved(a) == 20MiB
    purge!(a)
    @test reserved(a) == 0

    # when the release can't be followed by an event, the memory is still returned
    x = allocate_on(a, 3MiB, s)
    release_on(a, x, s)
    synchronize()
    inject_fault!(a, :release_event)
    @test_logs (:error, r"Failed to return GPU memory") purge!(a)
    @test reserved(a) == 0

    # a segment whose release failed is kept, as it may have been freed partially
    x = allocate_on(a, 3MiB, s)
    release_on(a, x, s)
    synchronize()
    inject_fault!(a, :release)
    @test_logs (:error, r"Failed to return GPU memory") purge!(a)
    @test reserved(a) == 20MiB
    # (and never reused)
    y = allocate_on(a, 3MiB, s)
    @test !overlaps(x, y)
    release_on(a, y, s)
    synchronize()
    purge!(a)
    @test reserved(a) == 20MiB
end

@testset "finishing a release while it is submitted" begin
    a = CachingAllocator(device())
    set_retention!(a, 0)
    s = stream()
    x = allocate_on(a, 4MiB, s)
    release_on(a, x, s)
    synchronize()
    # a task that releases segments, waiting before it frees them
    a.release_barrier = Base.Event()
    releasing = Threads.@spawn trim!(a)
    t0 = time()
    while CUDACore.queued_bytes(a) == 0 && time() - t0 < 60
        yield()
    end
    @test CUDACore.queued_bytes(a) == 20MiB
    # meanwhile, finishing releases leaves that one alone
    CUDACore.finish_releases!(a; blocking=true)
    @test reserved(a) == 20MiB
    notify(a.release_barrier)
    wait(releasing)
    a.release_barrier = nothing
    purge!(a)
    @test reserved(a) == 0
end

@testset "hard limit" begin
    a = CachingAllocator(device(); limits=(soft=UInt(0), hard=UInt(64MiB)))
    s = stream()
    xs = [allocate_on(a, 12MiB, s) for _ in 1:4]
    @test all(!isnothing, xs)
    @test reserved(a) == 48MiB
    # a medium segment doesn't fit anymore, but one of the requested size does
    y = allocate_on(a, 8MiB, s)
    @test y !== nothing
    @test reserved(a) <= 64MiB
    @test allocate_on(a, 12MiB, s) === nothing
    # freeing memory makes room, also when it's still in use by the GPU
    release_on(a, xs[1], s)
    z = allocate_on(a, 12MiB, s)
    @test z !== nothing
    @test reserved(a) <= 64MiB
    foreach(m -> release_on(a, m, s), [xs[2:end]; y; z])
    synchronize()
    purge!(a)
    @test reserved(a) == 0
end


# compute-sanitizer only sees the allocations the caching allocator makes, i.e., segments,
# unless the blocks are annotated
@testset "compute-sanitizer annotations" begin
    sanitizer = Sys.which("compute-sanitizer")
    if sanitizer === nothing || sanitize
        @info "Skipping the compute-sanitizer tests (not found, or already running under it)"
    else
        function sanitized(code; tool="memcheck")
            script = tempname() * ".jl"
            write(script, "using CUDA\n" * code)
            cmd = `$sanitizer --tool=$tool --print-limit=10 $(Base.julia_cmd()) --project=$(Base.active_project()) $script`
            cmd = addenv(cmd, "JULIA_CUDA_MEMORY_POOL" => "caching")
            out = IOBuffer()
            run(pipeline(ignorestatus(cmd); stdout=out, stderr=out))
            return String(take!(out))
        end

        # valid uses: the 8-bit atomics that cover the padding at the end of an array,
        # reusing memory while work on its previous allocation is still queued, graphs
        out = sanitized("""
        function inc!(a)
            i = threadIdx().x
            i <= length(a) && (@inbounds a[i] += one(eltype(a)))
            return
        end
        function atomic_last!(a)
            CUDA.@atomic a[length(a)] += UInt8(1)
            return
        end
        # 8- and 16-bit atomics on the last element touch the padded word
        a = CUDA.zeros(UInt8, 13)
        @cuda atomic_last!(a)
        @assert Array(a)[13] == 1
        # reusing memory while work on its previous allocation is still queued
        x = CUDA.zeros(Float32, 1000)
        @cuda threads=1000 inc!(x)
        CUDA.unsafe_free!(x)
        y = CUDA.zeros(Float32, 1000)
        @cuda threads=1000 inc!(y)
        @assert all(==(1), Array(y))
        # graphs
        z = CUDA.zeros(Float32, 256)
        g = capture() do
            @cuda threads=256 inc!(z)
        end
        exec = instantiate(g); exec(); exec()
        @assert all(==(2), Array(z))
        println("VALID DONE")
        """)
        @test occursin("VALID DONE", out)
        @test occursin("ERROR SUMMARY: 0 errors", out)

        # accesses past the end of a block, within its segment
        out = sanitized("""
        function write_at!(a, i)
            unsafe_store!(pointer(a), 1f0, i)
            return
        end
        a = CuArray{Float32}(undef, 16)
        b = CuArray{Float32}(undef, 16)
        @cuda write_at!(a, 17)   # just past the end, within the segment
        synchronize()
        """)
        @test occursin("Invalid __global__ write", out)

        # accesses to a block that has been freed
        out = sanitized("""
        function write_at!(p, i)
            unsafe_store!(p, 1f0, i)
            return
        end
        a = CuArray{Float32}(undef, 16)
        b = CuArray{Float32}(undef, 16)
        p = pointer(a)
        CUDA.unsafe_free!(a)
        @cuda write_at!(p, 1)   # after freeing, within the segment
        synchronize()
        """)
        @test occursin("Invalid __global__ write", out)

        # a reused block is uninitialized
        out = sanitized("""
        function copy1!(dst, src)
            dst[1] = src[1]
            return
        end
        x = CUDA.ones(Float32, 16)
        CUDA.unsafe_free!(x)
        y = CuArray{Float32}(undef, 16)   # reuses x's block
        out = CUDA.zeros(Float32, 1)
        @cuda copy1!(out, y)
        synchronize()
        """; tool="initcheck")
        @test occursin("Uninitialized __global__ memory read", out)
    end
end
