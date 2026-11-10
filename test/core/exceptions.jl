# XXX: these tests occasionally hang under compute-sanitizer
if !sanitize

# device-side exceptions leave the context usable, so they can be tested in this process.
# returns the diagnostics that the kernel printed.
function exception_output(f)
    mktemp() do path, io
        redirect_stdout(io) do
            @test_throws CUDACore.KernelException begin
                f()
                synchronize()

                # FIXME: on some platforms (Windows...), for some users, the exception flag
                # change doesn't immediately propagate to the host, and gets caught during
                # finalization. this looks like a driver bug, since we threadfence_system()
                # after setting the flag.
                # https://stackoverflow.com/questions/16417346/cuda-pinned-memory-flushing-from-the-device
                sleep(1)
                synchronize()
            end
            Base.Libc.flush_cstdio()
        end
        close(io)
        read(path, String)
    end
end

device_error_re = r"ERROR: a \w+ was thrown during kernel execution"

@testset "stack traces at different debug levels" begin

function store_kernel(arr, val)
    arr[threadIdx().x] = val
    return
end

@noinline foo(a, i) = a[1] = i
bar(a) = (foo(a, 42); nothing)

# debug level 0 only signals the exception
out = exception_output() do
    @cuda threads=3 debug_level=0 store_kernel(CuArray(zeros(Int)), 1)
end
@test !occursin(device_error_re, out)

# this process runs at the default debug level 1
out = exception_output() do
    @cuda threads=3 store_kernel(CuArray(zeros(Int)), 1)
end
@test count(device_error_re, out) == 1
@test count("BoundsError", out) == 1
@test count("Out-of-bounds array access", out) == 1
@test occursin("Stacktrace not available", out)

out = exception_output() do
    @cuda threads=3 debug_level=2 store_kernel(CuArray(zeros(Int)), 1)
end
@test count(device_error_re, out) == 1
@test count("BoundsError", out) == 1
@test count("Out-of-bounds array access", out) == 1
@test occursin(r"\] store_kernel at .*exceptions\.jl:\d+", out)

# JuliaGPU/CUDA.jl#329: frames of non-inlined calls
out = exception_output() do
    ptr = reinterpret(Core.LLVMPtr{Int,AS.Global}, C_NULL)
    arr = CuDeviceArray{Int,1,AS.Global}(ptr, (0,))
    @cuda debug_level=2 bar(arr)
end
@test occursin(device_error_re, out)
@test occursin(r"\] foo at .*exceptions\.jl:\d+", out)
@test occursin(r"\] bar at .*exceptions\.jl:\d+", out)

end

@testset "out-of-bounds indices" begin

function index_kernel(arr, I...)
    arr[I...] = 1
    return
end

# indices that main used to accept: a multidimensional index that linearizes into the
# array, and a non-positive linear index.
for I in ((3, 1), (0,))
    out = exception_output() do
        @cuda index_kernel(CuArray(zeros(Int, 2, 2)), I...)
    end
    @test count("BoundsError", out) == 1
end

end

@testset "bounds errors through generic checkbounds" begin
    # e.g. indexing a view uses Base's `checkbounds(::AbstractArray, I...)`
    function view_kernel(arr)
        view(arr, 1:1)[threadIdx().x] = 1
        return
    end

    out = exception_output() do
        @cuda threads=2 view_kernel(CuArray(zeros(Int, 2)))
    end
    @test count("BoundsError", out) == 1
    @test count("Out-of-bounds array access", out) == 1
end

@testset "Rational arithmetic" begin
    # on Julia 1.10 and 1.11, Base's helpers that throw for `Rational` overflow interpolate
    # the integer type into their message; GPUToolbox overrides them with constant messages.
    function rational_kernel(out, a, b, c)
        i = threadIdx().x
        out[i] = -muladd(a[i], b[i], c[i])
        return
    end
    a = [1//2, -3//4, 5//6]
    b = [2//3, 1//5, -7//8]
    c = [1//7, 2//9, 3//11]
    out = CuArray{Rational{Int}}(undef, 3)
    @cuda threads=3 rational_kernel(out, CuArray(a), CuArray(b), CuArray(c))
    @test Array(out) == -muladd.(a, b, c)
end

@testset "Rational exceptions" begin
    function rational_ctor_kernel(out, num, den)
        out[] = Rational(num, den)
        return
    end
    out = exception_output() do
        @cuda rational_ctor_kernel(CuArray([0//1]), 1, typemin(Int))
    end
    @test occursin(device_error_re, out)
    if isdefined(Base, :__throw_rational_argerror_typemin)   # Julia 1.10
        @test occursin("ArgumentError", out)
        @test occursin("invalid rational: denominator can't be typemin", out)
    else    # negating the denominator overflows
        @test occursin("OverflowError", out)
    end

    function rational_neg_kernel(out, x)
        out[] = -x
        return
    end
    out = exception_output() do
        @cuda rational_neg_kernel(CuArray([0//1]), typemin(Int)//1)
    end
    @test occursin(device_error_re, out)
    # Julia 1.12 builds the message lazily, so Base's method compiles and is only reported
    # as an exception of unknown type
    if VERSION < v"1.12-"
        @test occursin("OverflowError", out)
        @test occursin("rational numerator is typemin", out)
    end
end

@testset "exception output while unrelated work is running" begin
    function bad_kernel(a, i)
        a[i] = 1
        return
    end

    a = CUDA.zeros(Int, 1)
    worker = CuStream(; flags=CUDA.STREAM_NON_BLOCKING)
    # compile the kernel beforehand, as loading code waits for the GPU to become idle
    @cuda stream=worker bad_kernel(a, 1)
    synchronize(worker)

    for capturing in (false, true)
        out = mktemp() do path, io
            redirect_stdout(io) do
                while_gpu_busy() do
                    @cuda stream=worker bad_kernel(a, 2)
                    while !CUDACore.isdone(worker)
                        yield()
                    end
                    if capturing
                        # Stream queries themselves are prohibited during capture. Isolate
                        # the error reporting path after the failed kernel has already
                        # completed.
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
                end
            end
            close(io)
            read(path, String)
        end
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

if attribute(device(), CUDA.DEVICE_ATTRIBUTE_COMPUTE_MODE) != CUDA.COMPUTEMODE_EXCLUSIVE_PROCESS
    @testset "kernels launched in a destroyed context" begin
        destroyed = CuContext(device()) do ctx
            context!(ctx) do
                s = CuStream()
                @cuda stream=s identity(nothing)
                synchronize(s)
            end
            ctx
        end
        # (checks the exception flags of all contexts)
        @test synchronize() === nothing
        # the driver doesn't promise how it reports a destroyed context's handle, so it
        # shouldn't be checked again
        @test all(((ctx, _),) -> ctx != destroyed, CUDACore.exception_flags())
    end
end
