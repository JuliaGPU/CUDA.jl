# Per-test helpers, `include`d into each test's sandbox module. Kept separate
# from `setup.jl` (which runs once per worker and defines `CUDATestRecord`)
# so subpackage tests' own setup.jl can redefine `testf` without fighting a
# `using`-imported binding.

using CUDA, CUDACore, GPUArrays
using CUDA: i32
using Adapt
using FileCheck
using ..Main: TestSuite, can_use_cupti, sanitize, gpu_rss_nvml

testf(@nospecialize(f), @nospecialize(xs...); kwargs...) =
    TestSuite.compare(f, CuArray, xs...; kwargs...)

# NOTE: based on test/pkg.jl::capture_stdout, but doesn't discard exceptions
macro grab_output(ex)
    quote
        mktemp() do fname, fout
            ret = nothing
            open(fname, "w") do fout
                redirect_stdout(fout) do
                    ret = $(esc(ex))

                    # (synchronizing also flushes the output of kernels)
                    synchronize()
                end
            end
            ret, read(fname, String)
        end
    end
end

# Run some code on-device
macro on_device(ex...)
    code = ex[end]
    kwargs = ex[1:end-1]

    @gensym kernel
    esc(quote
        let
            function $kernel()
                $code
                return
            end

            CUDA.@sync @cuda $(kwargs...) $kernel()
        end
    end)
end

# helper function for sinking a value to prevent the callee from getting optimized away
@inline sink(i::Int32) =
    Base.llvmcall("""%slot = alloca i32
                     store volatile i32 %0, i32* %slot
                     %value = load volatile i32, i32* %slot
                     ret i32 %value""", Int32, Tuple{Int32}, i)
@inline sink(i::Int64) =
    Base.llvmcall("""%slot = alloca i64
                     store volatile i64 %0, i64* %slot
                     %value = load volatile i64, i64* %slot
                     ret i64 %value""", Int64, Tuple{Int64}, i)

function julia_exec(args::Cmd, env...)
    # FIXME: this doesn't work when the compute mode is set to exclusive
    cmd = Base.julia_cmd()
    cmd = `$cmd --project=$(Base.active_project()) --color=no $args`

    out = Pipe()
    err = Pipe()
    proc = run(pipeline(addenv(cmd, env...), stdout=out, stderr=err), wait=false)
    close(out.in)
    close(err.in)
    wait(proc)
    proc, read(out, String), read(err, String)
end

# keep the GPU busy until the host opens a gate, so that tests can do things while a kernel
# is running without depending on timing. if the gate is not opened in time (e.g., because
# the host is blocked waiting for the GPU), the kernel gives up and records that it timed
# out, instead of hanging. `gate` points to three flags: (is open, timed out, started).
function gate_kernel(gate::Ptr{UInt32}, cycles)
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
gate_timeout() = UInt64(60_000 * attribute(device(), CUDA.DEVICE_ATTRIBUTE_CLOCK_RATE))

# a gate that isn't opened in time usually means that a task blocked its thread in a call
# that waits for the GPU, without yielding to the task that opens the gate. to identify that
# call, have Julia print the native backtrace of every thread (as it does on SIGUSR1) when
# `f` takes longer than `timeout` seconds, i.e., before the gate kernel gives up. a POSIX
# timer delivers the signal, so this doesn't need any Julia thread to be responsive.
function with_watchdog(f; timeout=30)
    Sys.islinux() || return f()

    # the signal has to reach Julia's signal listener thread (the one waiting in
    # sigtimedwait), because Julia treats SIGUSR1 as fatal on threads that don't block it,
    # like the ones compute-sanitizer starts. for the same reason, make those threads ignore
    # it (Julia's own threads block it), as the profile that Julia collects after printing
    # the backtraces signals the whole process.
    tids = readdir("/proc/self/task")
    i = findfirst(tids) do tid
        wchan = try read("/proc/self/task/$tid/wchan", String) catch; "" end
        contains(wchan, "sigtimedwait")
    end
    i === nothing && return f()
    ccall(:signal, Ptr{Cvoid}, (Cint, Ptr{Cvoid}), 10 #= SIGUSR1 =#, Ptr{Cvoid}(1) #= SIG_IGN =#)

    # struct sigevent: sigev_value, sigev_signo, sigev_notify, and the target thread
    sev = zeros(Cint, 16)
    sev[3] = 10         # SIGUSR1
    sev[4] = 4          # SIGEV_THREAD_ID
    sev[5] = parse(Cint, tids[i])
    timer = Ref{Ptr{Cvoid}}()
    ccall((:timer_create, "librt.so.1"), Cint, (Cint, Ptr{Cint}, Ptr{Ptr{Cvoid}}),
          1 #= CLOCK_MONOTONIC =#, sev, timer) == 0 || return f()
    try
        # struct itimerspec: no interval, so the timer fires once
        spec = Int64[0, 0, timeout, 0]
        ccall((:timer_settime, "librt.so.1"), Cint,
              (Ptr{Cvoid}, Cint, Ptr{Int64}, Ptr{Cvoid}), timer[], 0, spec, C_NULL)
        f()
    finally
        ccall((:timer_delete, "librt.so.1"), Cint, (Ptr{Cvoid},), timer[])
    end
end

# run `f` while a kernel on another stream keeps the GPU busy, testing that the kernel was
# running all along. everything `f` does on the GPU needs to have been compiled beforehand,
# as loading code waits for the GPU to become idle.
function while_gpu_busy(f)
    gate = CuVector{UInt32,CUDA.HostMemory}(undef, 3)
    GC.@preserve gate begin
        cpu = pointer(gate; type=CUDA.HostMemory)
        gpu = reinterpret(Ptr{UInt32}, pointer(gate))
        # a blocking stream, so that waiting for the legacy stream or the device times out
        busy = CuStream()
        unsafe_store!(cpu, UInt32(1))
        @cuda stream=busy gate_kernel(gpu, gate_timeout())
        synchronize(busy)

        # freeing memory can also wait for the GPU, so don't run finalizers
        GC.gc(true)
        gc_enabled = GC.enable(false)
        with_watchdog() do
            try
                for i in 1:3
                    unsafe_store!(cpu, UInt32(0), i)
                end
                @cuda stream=busy gate_kernel(gpu, gate_timeout())
                t0 = time()
                while unsafe_load(cpu + 2sizeof(UInt32), :acquire) == 0 && time() - t0 < 60
                end
                @test unsafe_load(cpu, 3) == 1
                f()
            finally
                unsafe_store!(cpu, UInt32(1), :release)
                GC.enable(gc_enabled)
                synchronize(busy)
            end
        end
        @test unsafe_load(cpu, 2) == 0
    end
    return
end
