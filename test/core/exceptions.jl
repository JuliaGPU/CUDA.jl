# XXX: these tests occasionally hang under compute-sanitizer
if !sanitize

host_error_re = r"ERROR: (KernelException: exception thrown during kernel execution on device|CUDA error: an illegal instruction was encountered|CUDA error: unspecified launch failure)"
device_error_re = r"ERROR: a \w+ was thrown during kernel execution"

@testset "stack traces at different debug levels" begin

script = """
    using CUDA

    function kernel(arr, val)
        arr[threadIdx().x] = val
        return
    end

    cpu = zeros(Int)
    gpu = CuArray(cpu)
    @cuda threads=3 kernel(gpu, 1)
    synchronize()

    # FIXME: on some platforms (Windows...), for some users, the exception flag change
    # doesn't immediately propagate to the host, and gets caught during finalization.
    # this looks like a driver bug, since we threadfence_system() after setting the flag.
    # https://stackoverflow.com/questions/16417346/cuda-pinned-memory-flushing-from-the-device
    sleep(1)
    synchronize()
"""

# NOTE: kernel exceptions aren't always caught on the CPU as a KernelException.
#       on older devices, we emit a `trap` which causes a CUDA error...

let (proc, out, err) = julia_exec(`-g0 -e $script`)
    @test !success(proc)
    @test  occursin(host_error_re, err)
    @test !occursin(device_error_re, out)
    # NOTE: stdout sometimes contain a failure to free the CuArray with ILLEGAL_ACCESS
end

let (proc, out, err) = julia_exec(`-g1 -e $script`)
    @test !success(proc)
    @test occursin(host_error_re, err)
    @test count(device_error_re, out) == 1
    @test count("BoundsError", out) == 1
    @test count("Out-of-bounds array access", out) == 1
    @test occursin("Stacktrace not available", out)
end

let (proc, out, err) = julia_exec(`-g2 -e $script`)
    @test !success(proc)
    @test occursin(host_error_re, err)
    @test count(device_error_re, out) == 1
    @test count("BoundsError", out) == 1
    @test count("Out-of-bounds array access", out) == 1
    @test occursin("] kernel at $(joinpath(".", "none"))", out)
end

end

@testset "out-of-bounds indices" begin

# indices that main used to accept: a multidimensional index that linearizes into the
# array, and a non-positive linear index.
for index in ("3, 1", "0")
    script = """
        using CUDA

        function kernel(arr)
            arr[$index] = 1
            return
        end

        gpu = CuArray(zeros(Int, 2, 2))
        @cuda kernel(gpu)
        synchronize()

        # see "stack traces at different debug levels"
        sleep(1)
        synchronize()
    """

    let (proc, out, err) = julia_exec(`-g1 -e $script`)
        @test !success(proc)
        @test occursin(host_error_re, err)
        @test count("BoundsError", out) == 1
    end
end

end

@testset "bounds errors through generic checkbounds" begin
    # e.g. indexing a view uses Base's `checkbounds(::AbstractArray, I...)`
    script = """
        using CUDA

        function kernel(arr)
            view(arr, 1:1)[threadIdx().x] = 1
            return
        end

        gpu = CuArray(zeros(Int, 2))
        @cuda threads=2 kernel(gpu)
        synchronize()
        sleep(1)
        synchronize()
    """

    let (proc, out, err) = julia_exec(`-g1 -e $script`)
        @test !success(proc)
        @test occursin(host_error_re, err)
        @test count("BoundsError", out) == 1
        @test count("Out-of-bounds array access", out) == 1
    end
end

@testset "#329" begin

script = """
    using CUDA

    @noinline foo(a, i) = a[1] = i
    bar(a) = (foo(a, 42); nothing)

    ptr = reinterpret(Core.LLVMPtr{Int,AS.Global}, C_NULL)
    arr = CuDeviceArray{Int,1,AS.Global}(ptr, (0,))

    CUDA.@sync @cuda bar(arr)
"""

let (proc, out, err) = julia_exec(`-g2 -e $script`)
    @test !success(proc)
    @test occursin(host_error_re, err)
    @test occursin(device_error_re, out)
    @test occursin("foo at $(joinpath(".", "none"))", out)
    @test occursin("bar at $(joinpath(".", "none"))", out)
end

end

@testset "exception output while unrelated work is running" begin
for capturing in (false, true)
    script = """
        using CUDA, CUDACore, Test
        function gate_kernel(gate, cycles)
            unsafe_store!(gate, UInt32(1), 3)
            t0 = clock(UInt64)
            while unsafe_load(gate, :monotonic) == 0
                if clock(UInt64) - t0 >= cycles
                    unsafe_store!(gate, UInt32(1), 2)
                    break
                end
            end
            return
        end
        function bad_kernel(a, i)
            a[i] = 1
            return
        end
        gate = CuVector{UInt32,CUDA.HostMemory}(undef, 3)
        cpu = pointer(gate; type=CUDA.HostMemory)
        gpu = reinterpret(Ptr{UInt32}, pointer(gate))
        busy = CuStream()  # blocking stream: a context/default-stream wait would deadlock
        worker = CuStream(; flags=CUDA.STREAM_NON_BLOCKING)
        a = CUDA.zeros(Int, 1)
        timeout = UInt64(60_000 * attribute(device(), CUDA.DEVICE_ATTRIBUTE_CLOCK_RATE))
        unsafe_store!(cpu, UInt32(1))
        @cuda stream=busy gate_kernel(gpu, timeout)
        @cuda stream=worker bad_kernel(a, 1)
        synchronize(busy)
        synchronize(worker)
        GC.gc(true)
        enabled = GC.enable(false)
        try
            for i in 1:3
                unsafe_store!(cpu, UInt32(0), i)
            end
            @cuda stream=busy gate_kernel(gpu, timeout)
            t0 = time()
            while unsafe_load(cpu + 2sizeof(UInt32), :acquire) == 0 && time()-t0 < 60
            end
            @test unsafe_load(cpu, 3) == 1
            @cuda stream=worker bad_kernel(a, 2)
            while !CUDACore.isdone(worker)
                yield()
            end
            if $capturing
                # Stream queries themselves are prohibited during capture. Isolate the error
                # reporting path after the failed kernel has already completed.
                graph = capture() do
                    @test_throws CUDACore.KernelException CUDACore.check_exceptions()
                end
                @test graph isa CuGraph
            else
                @test_throws CUDACore.KernelException synchronize(worker)
            end
            Base.Libc.flush_cstdio()
            println("exception reported")
            flush(stdout)
            @test unsafe_load(cpu, 2) == 0
        finally
            unsafe_store!(cpu, UInt32(1))
            GC.enable(enabled)
            synchronize(busy)
        end
    """
    let (proc, out, err) = julia_exec(`-g1 -e $script`)
        @test success(proc)
        diagnostic = findfirst("BoundsError", out)
        reported = findfirst("exception reported", out)
        @test diagnostic !== nothing
        @test reported !== nothing
        if diagnostic !== nothing && reported !== nothing
            # A later synchronization during cleanup must not be what flushes the text.
            @test first(diagnostic) < first(reported)
        end
    end
end
end

end
