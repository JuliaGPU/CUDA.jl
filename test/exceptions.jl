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
"""

let (code, out, err) = julia_script(script, `-g0`)
    @test code == 1
    @test occursin("ERROR: KernelException: exception thrown during kernel execution on device", err)
    @test isempty(out)
end

let (code, out, err) = julia_script(script, `-g1`)
    @test code == 1
    @test occursin("ERROR: KernelException: exception thrown during kernel execution on device", err)
    @test occursin("ERROR: a exception was thrown during kernel execution", out)
    @test occursin("Run Julia on debug level 2 for device stack traces", out)
end

let (code, out, err) = julia_script(script, `-g2`)
    @test code == 1
    @test occursin("ERROR: KernelException: exception thrown during kernel execution on device", err)
    @test occursin("ERROR: a exception was thrown during kernel execution", out)
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

    ptr = CUDA.DevicePtr{Int,AS.Global}(0)
    arr = CuDeviceArray{Int,1,AS.Global}((0,), ptr)

    @cuda bar(arr)
    CUDA.synchronize()
"""

let (code, out, err) = julia_script(script, `-g2`)
    @test code == 1
    @test occursin("ERROR: KernelException: exception thrown during kernel execution on device", err)
    @test occursin("ERROR: a exception was thrown during kernel execution", out)
    @test occursin("foo at none:4", out)
    @test occursin("bar at none:5", out)
end

end
