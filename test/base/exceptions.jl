# these tests spawn subprocesses, so reset the current context to conserve memory
device_reset!()

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
    @test occursin("] checkbounds at $(joinpath(".", "abstractarray.jl"))", out)
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
    @test occursin(device_error_re, out)
    @test occursin("foo at $(joinpath(".", "none"))", out)
    @test occursin("bar at $(joinpath(".", "none"))", out)
end

end
