using CUDA: HostcallException

# the kernel state is part of the launch ABI and gets forwarded to dynamic parallelism
# child launches, so it must stay compact
@test sizeof(CUDACore.KernelState) == 2sizeof(UInt)

# Closed type arguments use TypeEgal dispatch keys on Julia 1.14 and later.
let K = Tuple{typeof(identity),Int,Tuple{Int}}
    P = @static if isdefined(Core, :TypeEgal)
        Core.TypeEgal{K}
    else
        Type{K}
    end
    @test CUDACore.hostcall_key_type(P) === K
end

hostcall_lookup(i::Int) = 10f0 * i
hostcall_increment(i::Int) = i + 1
const hostcall_counter = Threads.Atomic{Int}(0)
hostcall_count(i::Int) = (Threads.atomic_add!(hostcall_counter, i); nothing)
const hostcall_async_big_calls = Threads.Atomic{Int}(0)
function hostcall_async_big(i::Int)
    Threads.atomic_add!(hostcall_async_big_calls, 1)
    return ntuple(j -> i + j, 20)
end
hostcall_big(t::NTuple{12,Float64}, i::Int) =
    ntuple(j -> Int(t[mod1(j, 12)]) + i + j, 20)
hostcall_sum_ptr(ptr::CuPtr{Int}, n::Int) =
    Int(sum(Array(unsafe_wrap(CuArray, ptr, n))))
hostcall_boom(i::Int) = i == 3 ? error("boom $i") : 2i
hostcall_tagged(i::Int, tag::Int) = i + tag
hostcall_fail(::Int) = error("boom")
hostcall_double(x::Int) = 2.0 * x

struct HostcallScale
    a::Float32
end
(s::HostcallScale)(x) = s.a * x

# Calls from kernels into host functions, serviced by a foreign server thread.
@testset "service" begin
    function lookup(out)
        i = Int(threadIdx().x)
        out[i] = @hostcall(hostcall_lookup(i)::Float32) + 1f0
        return
    end
    out = CUDA.zeros(Float32, 64)
    @cuda threads=64 lookup(out)
    synchronize()
    @test Array(out) == 10f0 .* (1:64) .+ 1

    # partial warps, several blocks
    out = CUDA.zeros(Float32, 64)
    @cuda threads=20 blocks=3 lookup(out)
    synchronize()
    @test Array(out)[1:20] == 10f0 .* (1:20) .+ 1

    # the functional form and the macro, with an explicit return type
    function lookup2(out)
        i = Int(threadIdx().x)
        out[i] = @hostcall(hostcall_lookup(i)::Float32) +
                 hostcall(hostcall_lookup, Float32, i)
        return
    end
    out = CUDA.zeros(Float32, 64)
    @cuda threads=64 lookup2(out)
    synchronize()
    @test Array(out) == 20f0 .* (1:64)

    # the kernel keeps making progress while the main thread is blocked in the driver
    out = CUDA.zeros(Float32, 64)
    @cuda threads=64 lookup(out)
    synchronize(; blocking=true)
    @test Array(out) == 10f0 .* (1:64) .+ 1

    # many warps contending for ports
    function increment(out)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        out[i] = @hostcall hostcall_increment(Int(i))::Int
        return
    end
    out = CUDA.zeros(Int, 256 * 64)
    @cuda threads=256 blocks=64 increment(out)
    synchronize()
    @test Array(out) == (1:256*64) .+ 1

    # asynchronous calls are complete when synchronize() returns
    hostcall_counter[] = 0
    function count()
        i = Int(threadIdx().x)
        CUDA.hostcall_async(hostcall_count, i)
        @hostcall async=true hostcall_count(i)
        return
    end
    @cuda threads=32 blocks=4 count()
    synchronize()
    @test hostcall_counter[] == 2 * 4 * sum(1:32)

    hostcall_counter[] = 0
    @cuda threads=32 count()
    event = CuEvent()
    record(event)
    synchronize(event)
    @test hostcall_counter[] == 2 * sum(1:32)

    # Asynchronous calls ignore the return value and never wait for result packets.
    hostcall_async_big_calls[] = 0
    function async_big()
        CUDA.hostcall_async(hostcall_async_big, Int(threadIdx().x))
        return
    end
    @cuda threads=32 async_big()
    synchronize()
    @test hostcall_async_big_calls[] == 32

    # arguments and results larger than a packet
    function big(out)
        i = Int(threadIdx().x)
        t = ntuple(j -> Float64(i * j), Val(12))
        r = @hostcall hostcall_big(t, i)::NTuple{20,Int}
        s = 0
        for j in 1:20
            s += r[j]
        end
        out[i] = s
        return
    end
    out = CUDA.zeros(Int, 40)
    @cuda threads=40 big(out)
    synchronize()
    @test Array(out) == [sum(ntuple(j -> Int(Float64(i * mod1(j, 12))) + i + j, 20)) for i in 1:40]

    # device pointers arrive as CuPtr
    function pointer_kernel(out, arr)
        out[1] = @hostcall hostcall_sum_ptr(pointer(arr), length(arr))::Int
        return
    end
    arr = CuArray(1:10)
    out = CUDA.zeros(Int, 1)
    @cuda pointer_kernel(out, arr)
    synchronize()
    @test Array(out)[1] == 55

    # Targets are identified by the address of their rooted key type, like Julia's invoke
    # references its callee: distinct keys cannot collide, identifiers never fall in the
    # builtin id space, and registration is idempotent.
    K1 = Tuple{typeof(identity),Int,Tuple{Int}}
    K2 = Tuple{typeof(abs),Int,Tuple{Int}}
    id1 = CUDACore.hostcall_target_word(K1)
    id2 = CUDACore.hostcall_target_word(K2)
    @test id1 != id2
    @test id1 >= CUDACore.HOSTCALL_BUILTIN_IDS
    @test id1 == UInt64(CUDACore.GPUCompiler.resolve_relocation_target(
                           CUDACore.GPUCompiler.JuliaValueRef(K1)))
    @test id1 == CUDACore.hostcall_target_word(K1)
    CUDACore.register_hostcall_targets!([K1])
    CUDACore.register_hostcall_targets!([K1])
    target = CUDACore.hostcall_target(id1)
    @test target !== nothing
    @test target.key === K1
end

# handler and kernel for the redefinition testset below: they must be globals, both for the
# redefinition to be a plain method replacement (redefining a local boxes the binding, which
# the kernel would then capture) and so the kernel need not be recompiled
hostcall_redef(x::Int) = x + 1
function hostcall_redef_kernel(out)
    out[1] = @hostcall hostcall_redef(1)::Int
    return
end

@testset "handler redefinition" begin
    # hostcalls behave like `invokelatest`: redefining the handler between launches takes
    # effect without recompiling the kernel, re-seeding the service fast path
    out = CUDA.zeros(Int, 1)
    @cuda hostcall_redef_kernel(out)
    synchronize()
    @test Array(out)[1] == 2

    @eval hostcall_redef(x::Int) = x + 41
    @cuda hostcall_redef_kernel(out)
    synchronize()
    @test Array(out)[1] == 42
end

@testset "idle backoff" begin
    # This used to call `usleep` unconditionally, which is unavailable on Windows.
    @test_nowarn CUDACore.hostcall_backoff()
end

@testset "exceptions" begin
    function double(out)
        i = Int(threadIdx().x)
        out[i] = @hostcall hostcall_boom(i)::Int
        return
    end
    out = CUDA.zeros(Int, 8)
    @cuda threads=8 double(out)
    err = try
        synchronize()
        nothing
    catch err
        err
    end
    @test err isa HostcallException
    @test err.error isa ErrorException && err.error.msg == "boom 3"
    @test occursin("boom 3", sprint(showerror, err))
    @test err.device == device()
    # the exception has been consumed
    synchronize()
end

@testset "process layouts" begin
    # the server thread does not depend on Julia's thread pools: hostcalls keep being
    # serviced while the main thread is blocked in the driver, even without any other
    # Julia threads
    script = """
        using CUDA, Test
        lookup_value(i::Int) = Float32(i)
        function lookup(out)
            i = Int(threadIdx().x)
            out[i] = @hostcall(lookup_value(i)::Float32) + 1f0
            return
        end
        out = CUDA.zeros(Float32, 64)
        @cuda threads=64 lookup(out)
        synchronize(; blocking=true)
        @test Array(out) == Float32.(2:65)
        println("OK")
    """
    # Julia 1.10 and 1.11 reject an explicit zero for the interactive pool,
    # although `-t 1` produces the same one-default, zero-interactive layout.
    thread_layouts = VERSION >= v"1.12" ? ["1,0", "1,1", "2,1"] : ["1", "1,1", "2,1"]
    for threads in thread_layouts
        proc, out, err = julia_exec(`-t $threads -e $script`)
        success(proc) || @error "hostcall subprocess failed" threads stdout=out stderr=err
        @test success(proc)
        @test occursin("OK", out)
    end
end

if length(devices()) > 1
@testset "multiple devices" begin
    function increment(out)
        i = Int(threadIdx().x)
        out[i] = @hostcall hostcall_increment(i)::Int
        return
    end
    results = []
    for dev in devices()
        device!(dev) do
            out = CUDA.zeros(Int, 32)
            @cuda threads=32 increment(out)
            push!(results, out)
        end
    end
    for (dev, out) in zip(devices(), results)
        device!(dev) do
            synchronize()
            @test Array(out) == (1:32) .+ 1
        end
    end

    # concurrent blocking calls from kernels on every device: the server interleaves
    # contexts, and results must not cross over between areas
    function tagged(out, tag)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        out[i] = @hostcall hostcall_tagged(Int(i), tag)::Int
        return
    end
    outs = Dict(dev => device!(dev) do
                    out = CUDA.zeros(Int, 256 * 16)
                    tag = 1_000_000 * (deviceid(dev) + 1)
                    @cuda threads=256 blocks=16 tagged(out, tag)
                    out
                end for dev in devices())
    for dev in devices()
        device!(dev) do
            synchronize()
            @test Array(outs[dev]) == (1:256*16) .+ 1_000_000 * (deviceid(dev) + 1)
        end
    end

    # exceptions are reported per context: a handler error on one device surfaces when
    # synchronizing that device, not another one
    devA, devB = collect(Iterators.take(devices(), 2))
    function fail(out)
        out[1] = @hostcall hostcall_fail(1)::Int
        return
    end
    device!(devB) do
        # a single lane, so that exactly one exception is recorded
        @cuda threads=1 fail(CUDA.zeros(Int, 1))
        CUDACore.cuStreamSynchronize(stream())   # wait for the kernel without checking
    end
    device!(devA) do
        synchronize()
    end
    err = device!(devB) do
        try
            synchronize()
            nothing
        catch err
            err
        end
    end
    @test err isa HostcallException
    @test err.device == devB
end
end

@testset "static targets" begin
    function kernel(out)
        i = Int(threadIdx().x)
        a = @hostcall hostcall_double(i)::Float64
        b = @hostcall HostcallScale(3f0)(i)::Float32
        offset = 10
        c = @hostcall (x -> x + offset)(i)::Int     # closure capturing an isbits value
        d = hostcall(hostcall_double, Float64, i)
        out[i] = a + b + c + d
        return
    end
    out = CUDA.zeros(Float64, 32)
    @cuda threads=32 kernel(out)
    synchronize()
    @test Array(out) == [2i + 3i + i + 10 + 2i for i in 1:32]

    # the compiler records the targets and marks the kernel
    k = @cuda launch=false kernel(out)
    @test k.hostcall

    # asynchronous calls and the print family, whose output is emitted at synchronization
    function printer()
        @hostcall async=true println("thread ", threadIdx().x)
        @hostcall print("!")::Nothing
        return
    end
    _, output = @grab_output begin
        @cuda threads=2 printer()
        synchronize()
    end
    @test occursin("thread 1", output)
    @test occursin("thread 2", output)
    @test count("!", output) == 2
end

@testset "graph capture" begin
    # kernels replayed from a graph are not armed; they are serviced by the heartbeat
    function kernel(out)
        i = Int(threadIdx().x)
        out[i] = @hostcall hostcall_double(i)::Float64
        return
    end
    out = CUDA.zeros(Float64, 32)
    for i in 1:3
        CUDA.@captured begin
            @cuda threads=32 kernel(out)
        end
        synchronize()
        @test Array(out) == 2.0 .* (1:32)
        out .= 0
    end
end

@testset "precompiled kernels" begin
    # a kernel compiled during package precompilation carries its hostcall targets along
    # with the cached image, so calling it in a fresh session works without recompilation
    mktempdir() do dir
        pkgdir = joinpath(dir, "HostcallPrecompTest")
        mkpath(joinpath(pkgdir, "src"))
        write(joinpath(pkgdir, "src", "HostcallPrecompTest.jl"), """
            module HostcallPrecompTest
            using CUDA
            triple(x::Int) = 3.0 * x
            function kernel(out)
                i = Int(threadIdx().x)
                out[i] = @hostcall triple(i)::Float64
                @hostcall async=true println("precompiled hostcall ", i)
                return
            end
            const TT = Tuple{CuDeviceVector{Float64,CUDA.AS.Global}}
            function run()
                out = CUDA.zeros(Float64, 4)
                @cuda threads=4 kernel(out)
                synchronize()
                return Array(out)
            end
            if ccall(:jl_generating_output, Cint, ()) != 0 && CUDA.functional()
                # compile (but do not launch) the kernel during precompilation
                cufunction(kernel, TT)
            end
            end
            """)
        script = """
            pushfirst!(LOAD_PATH, $(repr(dir)))
            using CUDA, HostcallPrecompTest
            using CUDACore: GPUCompiler, methodinstance, CompilerJob, compiler_config
            # the image, with its hostcall targets, must come from the package image
            job = CompilerJob(methodinstance(typeof(HostcallPrecompTest.kernel), HostcallPrecompTest.TT),
                              compiler_config(device()))
            res = GPUCompiler.cached_results(CUDACore.CUDACompilerResults, job)
            println("cached: ", res !== nothing && res.image !== nothing && res.hostcall &&
                                length(res.hostcall_targets) == 2)
            println("result: ", HostcallPrecompTest.run())
        """
        # first run precompiles the package, the second one is a fresh session
        for i in 1:2
            proc, out, err = julia_exec(`-e $script`)
            @test success(proc)
            # GPUCompiler's package-image cache is only available on Julia 1.11+.
            VERSION >= v"1.11" && @test occursin("cached: true", out)
            @test occursin("result: [3.0, 6.0, 9.0, 12.0]", out)
            @test occursin("precompiled hostcall 1", out)
        end
    end
end
