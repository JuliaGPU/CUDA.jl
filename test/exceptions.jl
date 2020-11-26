# these tests spawn subprocesses, so reset the current context to conserve memory
CUDA.device_reset!()

@testset "stack traces at different debug levels" begin

script = """
    function kernel(arr, val)
        arr[1] = val
        return
    end

    cpu = zeros(Int)
    gpu = CuArray(cpu)
    @cuda kernel(gpu, 1.2)
    Array(gpu)

    # FIXME: on some platforms (Windows...), for some users, the exception flag change
    # doesn't immediately propagate to the host, and gets caught during finalization.
    # this looks like a driver bug, since we threadfence_system() after setting the flag.
    # https://stackoverflow.com/questions/16417346/cuda-pinned-memory-flushing-from-the-device
    sleep(1)
    synchronize()
    Array(gpu)
"""

let (code, out, err) = julia_script(script, `-g0`)
    @test code == 1
    @test  occursin("ERROR: KernelException: exception thrown during kernel execution on device", err)
    @test !occursin(r"ERROR: a \w+ was thrown during kernel execution", out)
    # NOTE: stdout sometimes contain a failure to free the CuArray with ILLEGAL_ACCESS
end

let (code, out, err) = julia_script(script, `-g1`)
    @test code == 1
    @test occursin("ERROR: KernelException: exception thrown during kernel execution on device", err)
    @test occursin(r"ERROR: a \w+ was thrown during kernel execution", out)
    @test occursin("Run Julia on debug level 2 for device stack traces", out)
end

let (code, out, err) = julia_script(script, `-g2`)
    @test code == 1
    @test occursin("ERROR: KernelException: exception thrown during kernel execution on device", err)
    @test occursin(r"ERROR: a \w+ was thrown during kernel execution", out)
    if VERSION < v"1.3.0-DEV.270"
        @test occursin("[1] Type at float.jl", out)
    else
        @test occursin("[1] Int64 at float.jl", out)
    end
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

let (code, out, err) = julia_script(script, `-g2`)
    @test code == 1
    @test occursin("ERROR: KernelException: exception thrown during kernel execution on device", err)
    @test occursin(r"ERROR: a \w+ was thrown during kernel execution", out)
    @test occursin("foo at none:4", out)
    @test occursin("bar at none:5", out)
end

end
