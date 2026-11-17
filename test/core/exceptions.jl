# XXX: these tests occasionally hang under compute-sanitizer
if !sanitize

# device-side exceptions leave the context usable, so they can be tested in this process.
# returns the exception that was reported.
function kernel_exception(f)
    err = try
        f()
        synchronize()

        # FIXME: on some platforms (Windows...), for some users, the exception flag
        # change doesn't immediately propagate to the host, and gets caught during
        # finalization. this looks like a driver bug, since we threadfence_system()
        # after setting the flag.
        # https://stackoverflow.com/questions/16417346/cuda-pinned-memory-flushing-from-the-device
        sleep(1)
        synchronize()
        nothing
    catch err
        err
    end
    @test err isa CUDACore.KernelException
    return err
end

@testset "reports at different debug levels" begin

function store_kernel(arr, val)
    arr[threadIdx().x] = val
    return
end

@noinline foo(a, i) = a[1] = i
bar(a) = (foo(a, 42); nothing)

# only one thread throws
function position_kernel(arr)
    i = threadIdx().y == 2 && blockIdx().z == 3 ? 2 : 1
    arr[i] = 1
    return
end

# debug level 0 only reports which thread threw
err = kernel_exception() do
    @cuda threads=(1, 2) blocks=(1, 1, 3) debug_level=0 position_kernel(CuArray([0]))
end
@test err.site === nothing
@test (err.thread, err.block) == ((1, 2, 1), (1, 1, 3))
@test startswith(sprint(showerror, err),
                 "KernelException: exception thrown during kernel execution on thread (1, 2, 1) in block (1, 1, 3)")

err = kernel_exception() do
    @cuda threads=(1, 2) blocks=(1, 1, 3) position_kernel(CuArray([0]))
end
@test err.site.name == "BoundsError"
@test (err.thread, err.block) == ((1, 2, 1), (1, 1, 3))

# this process runs at the default debug level 1
err = kernel_exception() do
    @cuda threads=3 blocks=2 store_kernel(CuArray(zeros(Int)), 1)
end
@test err.site.name == "BoundsError"
@test err.site.reason == "Out-of-bounds array access"
@test isempty(err.site.frames)
@test err.thread[1] in (2, 3) && err.thread[2:3] == (1, 1)
@test err.block[1] in (1, 2) && err.block[2:3] == (1, 1)
out = sprint(showerror, err)
@test occursin(r"^KernelException: BoundsError during kernel execution on thread \([23], 1, 1\) in block \([12], 1, 1\)", out)
@test occursin("Out-of-bounds array access", out)
@test occursin("Stacktrace not available", out)

err = kernel_exception() do
    @cuda threads=3 debug_level=2 store_kernel(CuArray(zeros(Int)), 1)
end
@test err.site.name == "BoundsError"
@test err.site.reason == "Out-of-bounds array access"
@test any(frame -> frame.func == "store_kernel" && endswith(frame.file, "exceptions.jl"),
          err.site.frames)
@test occursin(r"\] store_kernel at .*exceptions\.jl:\d+", sprint(showerror, err))

# JuliaGPU/CUDA.jl#329: frames of non-inlined calls
err = kernel_exception() do
    ptr = reinterpret(Core.LLVMPtr{Int,AS.Global}, C_NULL)
    arr = CuDeviceArray{Int,1,AS.Global}(ptr, (0,))
    @cuda debug_level=2 bar(arr)
end
out = sprint(showerror, err)
@test occursin(r"\] foo at .*exceptions\.jl:\d+", out)
@test occursin(r"\] bar at .*exceptions\.jl:\d+", out)

end

@testset "throw sites of several kernels" begin
    # every kernel has a table of its own, and the host looks up the one of the kernel that
    # threw
    function sqrt_kernel(a)
        a[1] = sqrt(a[1])
        return
    end
    function index_kernel(a, i)
        a[i] = 1
        return
    end
    a = CuArray([1.0])
    @cuda sqrt_kernel(a)
    @cuda index_kernel(a, 1)
    synchronize()

    err = kernel_exception() do
        @cuda index_kernel(a, 2)
    end
    @test err.site.name == "BoundsError"
    err = kernel_exception() do
        a .= -1
        @cuda sqrt_kernel(a)
    end
    @test err.site.name == "DomainError"
end

@testset "out-of-bounds indices" begin

function index_kernel(arr, I...)
    arr[I...] = 1
    return
end

# indices that main used to accept: a multidimensional index that linearizes into the
# array, and a non-positive linear index.
for I in ((3, 1), (0,))
    err = kernel_exception() do
        @cuda index_kernel(CuArray(zeros(Int, 2, 2)), I...)
    end
    @test err.site.name == "BoundsError"
end

end

@testset "bounds errors through generic checkbounds" begin
    # e.g. indexing a view uses Base's `checkbounds(::AbstractArray, I...)`
    function view_kernel(arr)
        view(arr, 1:1)[threadIdx().x] = 1
        return
    end

    err = kernel_exception() do
        @cuda threads=2 view_kernel(CuArray(zeros(Int, 2)))
    end
    @test err.site.name == "BoundsError"
    @test err.site.reason == "Out-of-bounds array access"
    @test err.thread == (2, 1, 1)
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
    err = kernel_exception() do
        @cuda rational_ctor_kernel(CuArray([0//1]), 1, typemin(Int))
    end
    if isdefined(Base, :__throw_rational_argerror_typemin)   # Julia 1.10
        @test err.site.name == "ArgumentError"
        @test err.site.reason == "invalid rational: denominator can't be typemin"
    else    # negating the denominator overflows
        @test err.site.name == "OverflowError"
    end

    function rational_neg_kernel(out, x)
        out[] = -x
        return
    end
    err = kernel_exception() do
        @cuda rational_neg_kernel(CuArray([0//1]), typemin(Int)//1)
    end
    # Julia 1.12 builds the message lazily, so Base's method compiles and is only reported
    # as an exception of unknown type
    if VERSION < v"1.12-"
        @test err.site.name == "OverflowError"
        @test err.site.reason == "rational numerator is typemin"
    end
end

@testset "exceptions while unrelated work is running" begin
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
        while_gpu_busy() do
            @cuda stream=worker bad_kernel(a, 2)
            while !CUDACore.isdone(worker)
                yield()
            end
            err = nothing
            if capturing
                # Stream queries themselves are prohibited during capture. Isolate the
                # error reporting path after the failed kernel has already completed.
                graph = capture() do
                    err = try
                        CUDACore.check_exceptions()
                    catch err
                        err
                    end
                end
                @test graph isa CuGraph
            else
                err = try
                    synchronize(worker)
                catch err
                    err
                end
            end
            @test err isa CUDACore.KernelException
            @test err.site.name == "BoundsError"
        end
    end
end

@testset "exceptions of a kernel that ran before another one was loaded" begin
    function failing_kernel(a)
        a[2] = 1
        return
    end
    loaded_later() = return

    # loading the module of a new kernel reports the exception, which isn't a problem with
    # the new kernel's image
    logger = Test.TestLogger(; min_level=Base.CoreLogging.Warn)
    err = kernel_exception() do
        @cuda failing_kernel(CuArray([0]))
        while !CUDACore.isdone(stream())
            yield()
        end
        Base.CoreLogging.with_logger(logger) do
            @cuda loaded_later()
        end
    end
    @test err.site.name == "BoundsError"
    @test isempty(logger.logs)
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
