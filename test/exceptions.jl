# NVIDIA bug 3263616: compute-sanitizer crashes when generating host backtraces,
#                     but --show-backtrace=no does not survive execve.
@not_if_sanitize begin

# these tests spawn subprocesses, so reset the current context to conserve memory
CUDA.can_reset_device() && device_reset!()

host_error_re = r"ERROR: (KernelException: exception thrown during kernel execution on device|CUDA error: an illegal instruction was encountered|CUDA error: unspecified launch failure)"
device_error_re = r"ERROR: a \w+ was thrown during kernel execution"

@testset "stack traces at different debug levels" begin

script = """
    function kernel(arr, val)
        arr[1] = val
        return
    end

    cpu = zeros(Int)
    gpu = CuArray(cpu)
    @cuda kernel(gpu, 1.2)
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
#

let (code, out, err) = julia_script(script, `-g0`)
    @test code == 1
    @test  occursin(host_error_re, err)
    @test !occursin(device_error_re, out)
    # NOTE: stdout sometimes contain a failure to free the CuArray with ILLEGAL_ACCESS
end

let (code, out, err) = julia_script(script, `-g1`)
    @test code == 1
    @test occursin(host_error_re, err)
    @test occursin(device_error_re, out)
    @test occursin("Run Julia on debug level 2 for device stack traces", out)
end

let (code, out, err) = julia_script(script, `-g2`,
                                    "JULIA_CUDA_DEBUG_INFO"=>false) # NVIDIA#3305774
    @test code == 1
    @test occursin(host_error_re, err)
    @test occursin(device_error_re, out)
    @test occursin("[1] Int64 at float.jl", out)
    @test occursin("[4] kernel at none:5", out)
end

end

@testset "#329" begin

script = """
    @noinline foo(a, i) = a[1] = i
    bar(a) = (foo(a, 42); nothing)

    ptr = reinterpret(Core.LLVMPtr{Int,AS.Global}, C_NULL)
    arr = CuDeviceArray{Int,1,AS.Global}((0,), ptr)

    CUDA.@sync @cuda bar(arr)
"""

let (code, out, err) = julia_script(script, `-g2`,
                                    "JULIA_CUDA_DEBUG_INFO"=>false) # NVIDIA#3305774
    @test code == 1
    @test occursin(host_error_re, err)
    @test occursin(device_error_re, out)
    @test occursin("foo at none:4", out)
    @test occursin("bar at none:5", out)
end

end

end
