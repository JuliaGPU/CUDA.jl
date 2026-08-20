using CUDA: HostFunction, HostcallException

# the hostcall service: calls from kernels into host functions, serviced by a foreign
# server thread. these tests use runtime-registered handles; statically-known targets are
# tested separately.

@testset "handles" begin
    data = Float32.(1:64) .* 10
    hf = HostFunction(Float32, Tuple{Int}) do i
        data[i]
    end
    @test hf.id & CUDACore.HOSTCALL_RUNTIME_ID_BIT != 0

    function lookup(out, hf)
        i = threadIdx().x
        out[i] = hf(i) + 1f0
        return
    end
    out = CUDA.zeros(Float32, 64)
    @cuda threads=64 lookup(out, hf)
    synchronize()
    @test Array(out) == data .+ 1

    # partial warps, several blocks
    out = CUDA.zeros(Float32, 64)
    @cuda threads=20 blocks=3 lookup(out, hf)
    synchronize()
    @test Array(out)[1:20] == data[1:20] .+ 1

    # the functional form and the macro, with an explicit return type
    function lookup2(out, hf)
        i = threadIdx().x
        out[i] = @hostcall(hf(i)::Float32) + hostcall(hf, Float32, i)
        return
    end
    out = CUDA.zeros(Float32, 64)
    @cuda threads=64 lookup2(out, hf)
    synchronize()
    @test Array(out) == 2 .* data

    # the kernel keeps making progress while the main thread is blocked in the driver
    out = CUDA.zeros(Float32, 64)
    @cuda threads=64 lookup(out, hf)
    synchronize(; blocking=true)
    @test Array(out) == data .+ 1

    # many warps contending for ports
    hf_inc = HostFunction(Int, Tuple{Int}) do i
        i + 1
    end
    function increment(out, hf)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        out[i] = hf(Int(i))
        return
    end
    out = CUDA.zeros(Int, 256 * 64)
    @cuda threads=256 blocks=64 increment(out, hf_inc)
    synchronize()
    @test Array(out) == (1:256*64) .+ 1

    # asynchronous calls are complete when synchronize() returns
    counter = Threads.Atomic{Int}(0)
    hf_count = HostFunction(Nothing, Tuple{Int}) do i
        Threads.atomic_add!(counter, i)
        nothing
    end
    function count(hf)
        i = Int(threadIdx().x)
        CUDA.hostcall_async(hf, i)
        @hostcall async=true hf(i)
        return
    end
    @cuda threads=32 blocks=4 count(hf_count)
    synchronize()
    @test counter[] == 2 * 4 * sum(1:32)

    counter[] = 0
    @cuda threads=32 count(hf_count)
    event = CuEvent()
    record(event)
    synchronize(event)
    @test counter[] == 2 * sum(1:32)

    # An asynchronous runtime handle has no reply protocol, regardless of its declared
    # blocking return type. In particular, large returns must not wait for more packets.
    async_big_calls = Threads.Atomic{Int}(0)
    hf_async_big = HostFunction(NTuple{20,Int}, Tuple{Int}) do i
        Threads.atomic_add!(async_big_calls, 1)
        ntuple(j -> i + j, 20)
    end
    function async_big(hf)
        CUDA.hostcall_async(hf, Int(threadIdx().x))
        return
    end
    @cuda threads=32 async_big(hf_async_big)
    synchronize()
    @test async_big_calls[] == 32

    # arguments and results larger than a packet
    hf_big = HostFunction(NTuple{20,Int}, Tuple{NTuple{12,Float64}, Int}) do t, i
        ntuple(j -> Int(t[mod1(j, 12)]) + i + j, 20)
    end
    function big(out, hf)
        i = Int(threadIdx().x)
        t = ntuple(j -> Float64(i * j), Val(12))
        r = hf(t, i)
        s = 0
        for j in 1:20
            s += r[j]
        end
        out[i] = s
        return
    end
    out = CUDA.zeros(Int, 40)
    @cuda threads=40 big(out, hf_big)
    synchronize()
    @test Array(out) == [sum(ntuple(j -> Int(Float64(i * mod1(j, 12))) + i + j, 20)) for i in 1:40]

    # device pointers arrive as CuPtr
    hf_ptr = HostFunction(Int, Tuple{CuPtr{Int}, Int}) do ptr, n
        # handlers may use the CUDA API on the hostcall stream, but must not compile or
        # load kernels (module loading synchronizes the device), so copy to the host
        Int(sum(Array(unsafe_wrap(CuArray, ptr, n))))
    end
    function pointer_kernel(out, arr, hf)
        out[1] = hf(pointer(arr), length(arr))
        return
    end
    arr = CuArray(1:10)
    out = CUDA.zeros(Int, 1)
    @cuda pointer_kernel(out, arr, hf_ptr)
    synchronize()
    @test Array(out)[1] == 55

    # argument types are validated
    @test_throws ArgumentError HostFunction(identity, String, Tuple{Int})
    @test_throws ArgumentError HostFunction(identity, Int, Tuple{String})

    # A hash collision must fail at registration instead of silently changing dispatch.
    K1 = Tuple{typeof(identity),Int,Tuple{Int}}
    K2 = Tuple{typeof(abs),Int,Tuple{Int}}
    id = CUDACore.hostcall_target_id_value(K1)
    CUDACore.register_hostcall_targets!([id => K1])
    @test_throws ErrorException CUDACore.register_hostcall_targets!([id => K2])

    # The registry keeps a target alive after the Julia handle itself becomes unreachable;
    # only an explicit close may invalidate a device handle.
    function temporary_handle(out)
        hf = HostFunction(x -> x + 1, Int, Tuple{Int})
        @cuda threads=1 lookup(out, hf)
        return nothing
    end
    out = CUDA.zeros(Float32, 1)
    temporary_handle(out)
    GC.gc()
    synchronize()
    @test Array(out) == [3]
end

@testset "exceptions" begin
    hf = HostFunction(Int, Tuple{Int}) do i
        i == 3 && error("boom $i")
        i * 2
    end
    function double(out, hf)
        out[threadIdx().x] = hf(Int(threadIdx().x))
        return
    end
    out = CUDA.zeros(Int, 8)
    @cuda threads=8 double(out, hf)
    err = try
        synchronize()
        nothing
    catch err
        err
    end
    @test err isa HostcallException
    @test err.error isa ErrorException && err.error.msg == "boom 3"
    @test occursin("boom 3", sprint(showerror, err))
    # the exception has been consumed
    synchronize()

    # calling a closed handle is an error
    hf2 = HostFunction(NTuple{20,Int}, Tuple{Int}) do i
        ntuple(j -> i + j, 20)
    end
    close(hf2)
    function closed_handle(out, hf)
        out[1] = sum(hf(1))
        return
    end
    @cuda closed_handle(out, hf2)
    @test_throws HostcallException synchronize()
end

@testset "process layouts" begin
    # the server thread does not depend on Julia's thread pools: hostcalls keep being
    # serviced while the main thread is blocked in the driver, even without any other
    # Julia threads
    script = """
        using CUDA, Test
        data = Float32.(1:64)
        hf = CUDA.HostFunction(Float32, Tuple{Int}) do i
            data[i]
        end
        function lookup(out, hf)
            i = threadIdx().x
            out[i] = hf(i) + 1f0
            return
        end
        out = CUDA.zeros(Float32, 64)
        @cuda threads=64 lookup(out, hf)
        synchronize(; blocking=true)
        @test Array(out) == data .+ 1
        println("OK")
    """
    for threads in ["1,0", "1,1", "2,1"]
        proc, out, err = julia_exec(`-t $threads -e $script`)
        @test success(proc)
        @test occursin("OK", out)
    end
end

if length(devices()) > 1
@testset "multiple devices" begin
    hf = HostFunction(Int, Tuple{Int}) do i
        i + 1
    end
    function increment(out, hf)
        out[threadIdx().x] = hf(Int(threadIdx().x))
        return
    end
    results = []
    for dev in devices()
        device!(dev) do
            out = CUDA.zeros(Int, 32)
            @cuda threads=32 increment(out, hf)
            push!(results, out)
        end
    end
    for (dev, out) in zip(devices(), results)
        device!(dev) do
            synchronize()
            @test Array(out) == (1:32) .+ 1
        end
    end
end
end
