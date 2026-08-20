# XXX: these tests occasionally hang under compute-sanitizer
if !sanitize

# with hostcall available (the default), the device's report travels with the exception;
# without it, the device prints it with printf and the exception is message-less
host_error_re = r"ERROR: (KernelException: .*during kernel execution on device|CUDA error: an illegal instruction was encountered|CUDA error: unspecified launch failure)"
device_error_re = r"a \w+ was thrown during kernel execution"

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
    @test !occursin(device_error_re, err)
    # NOTE: stdout sometimes contain a failure to free the CuArray with ILLEGAL_ACCESS
end

let (proc, out, err) = julia_exec(`-g1 -e $script`)
    @test !success(proc)
    @test occursin(host_error_re, err)
    @test count(device_error_re, err) == 1
    @test count("BoundsError", err) == 1
    @test count("Out-of-bounds array access", err) == 1
    @test occursin("Stacktrace not available", err)
    @test !occursin(device_error_re, out)
end

let (proc, out, err) = julia_exec(`-g2 -e $script`)
    @test !success(proc)
    @test occursin(host_error_re, err)
    @test count(device_error_re, err) == 1
    @test count("BoundsError", err) == 1
    @test count("Out-of-bounds array access", err) == 1
    @test occursin("] kernel at $(joinpath(".", "none"))", err)
    @test !occursin(device_error_re, out)
end

# without hostcall, the device prints the report itself
let (proc, out, err) = julia_exec(`-g2 -e $script`, "JULIA_CUDA_HOSTCALL" => "false")
    @test !success(proc)
    @test occursin(host_error_re, err)
    @test count(device_error_re, out) == 1
    @test count("BoundsError", out) == 1
    @test count("Out-of-bounds array access", out) == 1
    @test occursin("] kernel at $(joinpath(".", "none"))", out)
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
    @test occursin(device_error_re, err)
    @test occursin("foo at $(joinpath(".", "none"))", err)
    @test occursin("bar at $(joinpath(".", "none"))", err)
end

end

@testset "in-process" begin
    # the exception carries the device's report
    function kernel(arr, val)
        arr[threadIdx().x] = val
        return
    end
    gpu = CuArray(zeros(Int))
    @cuda threads=3 kernel(gpu, 1)
    err = try
        synchronize()
        nothing
    catch err
        err
    end
    @test err isa CUDACore.KernelException
    msg = sprint(showerror, err)
    @test occursin(device_error_re, msg)
    @test occursin("BoundsError", msg)
    @test occursin("Out-of-bounds array access", msg)
    # the exception has been consumed, and the device is usable again
    synchronize()
    gpu = CuArray(zeros(Int, 3))
    @cuda threads=3 kernel(gpu, 1)
    synchronize()
    @test Array(gpu) == [1, 1, 1]

    # a report from a kernel while the main thread is blocked in the driver
    gpu = CuArray(zeros(Int))
    @cuda threads=3 kernel(gpu, 1)
    err = try
        synchronize(; blocking=true)
        nothing
    catch err
        err
    end
    @test err isa CUDACore.KernelException
    @test occursin("BoundsError", sprint(showerror, err))
    synchronize()
end

@testset "precompiled kernels" begin
    # exception reports do not depend on any per-kernel registration, so a kernel
    # compiled during precompilation reports just as well in a fresh session
    mktempdir() do dir
        pkgdir = joinpath(dir, "ExceptionPrecompTest")
        mkpath(joinpath(pkgdir, "src"))
        write(joinpath(pkgdir, "src", "ExceptionPrecompTest.jl"), """
            module ExceptionPrecompTest
            using CUDA
            function kernel(arr, val)
                arr[threadIdx().x] = val
                return
            end
            const TT = Tuple{CuDeviceVector{Int,CUDA.AS.Global}, Int}
            function run()
                gpu = CuArray(zeros(Int))
                @cuda threads=3 kernel(gpu, 1)
                synchronize()
            end
            if ccall(:jl_generating_output, Cint, ()) != 0 && CUDA.functional()
                cufunction(kernel, TT)
            end
            end
            """)
        script = """
            pushfirst!(LOAD_PATH, $(repr(dir)))
            using CUDA, ExceptionPrecompTest
            ExceptionPrecompTest.run()
        """
        for i in 1:2
            proc, out, err = julia_exec(`-g1 -e $script`)
            @test !success(proc)
            @test occursin(host_error_re, err)
            @test occursin("BoundsError", err)
        end
    end
end

end
