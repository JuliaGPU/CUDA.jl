import JSON
import NVTX
using CUDA.Profile: push_row!

# A100s let only one process profile them when MIG is enabled, which other CI jobs on the
# same GPU may be doing, so retry sessions that find the GPU busy
function retry_busy(f; timeout=600)
    start = time()
    while true
        try
            return f()
        catch err
            busy = occursin("being profiled by another process", sprint(showerror, err))
            busy && time() - start < timeout || rethrow()
            sleep(5)
        end
    end
end
macro retry_busy(ex)
    :(retry_busy(() -> $(esc(ex))))
end

# Retry contention outside the deadlock timeout: another process may hold the profiling
# hardware longer than the operation itself is allowed to take.
function bounded(f; timeout=120)
    try
        retry_busy() do
            t = Threads.@spawn f()
            timedwait(() -> istaskdone(t), timeout) == :ok ? fetch(t) : :timeout
        end
    catch err
        err
    end
end

# whether a finalizer runs within `session(body)`, which, being a profiling session, may
# hold locks. the object escapes, so that the compiler can't inline its finalizer.
const finalized = Ref(false)
const finalizable = Any[nothing]
@noinline function drop_finalizable()
    finalizable[1] = finalizer(_ -> (finalized[] = true), Ref(0))
    finalizable[1] = nothing
end
function finalizes_within(session)
    finalized[] = false
    seen = Ref(false)
    session() do
        drop_finalizable()
        GC.gc()
        seen[] = finalized[]
    end
    return seen[]
end

# profiler results with the given events, which take place between the synchronizations
# that delimit the profiled code (from 1 to 99 s). rows only need to specify some columns.
function synthetic_results(; host=[], device=[], nvtx=[], kwargs...)
    h = (id=Int[], start=Float64[], stop=Float64[], name=String[], tid=Int[],
         details=Union{Missing,String}[])
    sync(id, t) = (; id, start=t - 1, stop=t, name="cuCtxSynchronize", tid=1)
    for row in [sync(1, 1.0); host; sync(1000, 99.0)]
        push_row!(h, merge((; tid=1), row))
    end
    d = (id=Int[], start=Float64[], stop=Float64[], name=String[],
         device=Int[], context=Int[], stream=Int[],
         grid=Union{Missing,CUDACore.CuDim3}[], block=Union{Missing,CUDACore.CuDim3}[],
         registers=Union{Missing,Int64}[],
         shared_mem=Union{Missing,@NamedTuple{static::Int64,dynamic::Int64}}[],
         local_mem=Union{Missing,@NamedTuple{thread::Int64,total::Int64}}[],
         size=Union{Missing,Int64}[], details=Union{Missing,String}[])
    for row in device
        push_row!(d, merge((; name="kernel", device=0, context=1, stream=1), row))
    end
    n = (id=Int[], start=Float64[], type=Symbol[], tid=Int[],
         name=Union{Missing,String}[], domain=Union{Missing,String}[],
         payload=Any[], color=Union{Nothing,UInt32}[], category=Union{Missing,UInt32}[])
    for row in nvtx
        push_row!(n, merge((; tid=1, color=nothing), row))
    end
    CUDA.Profile.ProfileResults(; host=h, device=d, nvtx=n, kwargs...)
end

@testset "profiler" begin

############################################################################################

@testset "helpers" begin

@testset "contention" begin
    attempts = Ref(0)
    result = bounded(; timeout=1) do
        attempts[] += 1
        attempts[] == 1 && error("GPU is being profiled by another process")
        :done
    end
    @test result === :done
    @test attempts[] == 2

    # Other errors still reach the assertion without being retried.
    attempts[] = 0
    err = bounded() do
        attempts[] += 1
        error("profiled code failed")
    end
    @test occursin("profiled code failed", sprint(showerror, err))
    @test attempts[] == 1

    # A task that actually blocks still times out; release it afterwards.
    release, done = Channel{Nothing}(1), Channel{Nothing}(1)
    @test bounded(; timeout=1) do
        take!(release)
        put!(done, nothing)
    end === :timeout
    put!(release, nothing)
    take!(done)
end

@testset "native buffer callbacks" begin
    # CUPTI may call them while Julia collects garbage (#3536), so they only call `malloc`,
    # through a pointer, and use atomics for the state they share between threads
    LLVM = CUPTI.LLVM
    ctx = LLVM.Context(; opaque_pointers=true)
    try
        mod = parse(LLVM.Module, CUPTI.callback_ir)
        LLVM.verify(mod)
        @test sort([LLVM.name(f) for f in LLVM.functions(mod)]) == ["complete", "request"]
        insts = [inst for f in LLVM.functions(mod) for bb in LLVM.blocks(f)
                      for inst in LLVM.instructions(bb)]
        calls = filter(inst -> inst isa LLVM.CallInst, insts)
        @test length(calls) == 1
        @test !(LLVM.called_operand(only(calls)) isa LLVM.Function)
        @test any(inst -> inst isa LLVM.AtomicCmpXchgInst, insts)
        @test any(inst -> inst isa LLVM.AtomicRMWInst, insts)
    finally
        LLVM.dispose(ctx)
    end
end

@testset "detect_cupti" begin
    @test @inferred(CUDA.Profile.detect_cupti()) isa Bool
end

end

############################################################################################

@testset "external" begin

CUDA.Profile.start()
CUDA.Profile.stop()

@test CUDA.@profile external=true begin
    true
end

end

############################################################################################

if CUPTI.version() < v"11"
    @test_throws "Integrated profiling requires CUPTI 11" CUDA.@profile error("must not execute")
end

@static if can_use_cupti() && CUPTI.version() >= v"11"
@testset "integrated" begin

@testset "activity cleanup" begin
    @test_throws ErrorException CUDA.@profile error("profiled code failed")
    @test CUDA.@profile(identity(nothing)) isa CUDA.Profile.ProfileResults
end

@testset "many activity records" begin
    # more records than fit in one activity buffer
    results = CUDA.@profile for _ in 1:100_000
        @cuda identity(nothing)
    end
    @test count(contains("identity"), results.device.name) == 100_000
end

@testset "profiled allocation" begin
    a = nothing
    CUDA.@profile begin
        a = CuArray([1, 2, 3])
    end
    @test Array(a) == [1, 2, 3]
end

@testset "finalizers" begin
    # code that frees GPU memory relies on finalizers (#3530)
    @test finalizes_within(f -> CUDA.@profile f())
    # SASS reflection breaks later sessions with these versions of CUPTI: 2024.1.1 (CUDA
    # 12.4) reports activity of earlier sessions again, and 2024.2 and 2024.3 lose kernel
    # records (NVIDIA bug #4667039)
    if !(v"2024.1.1" <= CUPTI.library_version() < v"2025.1.0")
        @test finalizes_within(f -> CUDATools.code_sass(f, devnull))
    end
end

if CUDACore.is_tegra() && CUDA.runtime_version() < v"12"
    @testset "unsupported Tegra activity" begin
        cfg = CUPTI.ActivityConfig([CUPTI.CUPTI_ACTIVITY_KIND_DRIVER,
                                    CUPTI.CUPTI_ACTIVITY_KIND_MARKER_DATA])
        ran = Ref(false)
        CUPTI.@enable! cfg begin
            ran[] = true
            CUDA.synchronize()
        end
        @test ran[]
    end
end

# smoke test
let
    str = string(CUDA.@profile true)
    @test occursin("No host-side activity was recorded", str)
    @test occursin("No device-side activity was recorded", str)
end

# kernel launch
let
    @cuda identity(nothing)
    str = string(CUDA.@profile @cuda identity(nothing))

    @test occursin("cuLaunchKernel", str)
    @test occursin("identity()", str)

    @test !occursin("cuCtxSynchronize", str)
    @test !occursin("ID", str)

    # without cropping, rows aren't padded to the width of the table
    @test !any(line -> endswith(line, r"[ │┐┤┘]"), split(str, '\n'))
end

# kernel launch (trace)
let
    str = string(CUDA.@profile trace=true @cuda identity(nothing))

    @test occursin("cuLaunchKernel", str)
    @test occursin("identity()", str)

    @test occursin("ID", str)

    @test !occursin("cuCtxSynchronize", str)
end

# timestamps keep their nanosecond resolution, and aren't rounded to multiples of
# 2^-22 s (~238 ns) as happens when converting CUPTI's raw timestamps to seconds
let
    results = CUDA.@profile for i in 1:10
        @cuda identity(nothing)
    end
    times = [results.host.stop .- results.host.start; results.device.stop .- results.device.start]
    @test any(t -> !isinteger(t * 2^22), times)
end

# kernel launch (raw trace)
let
    str = string(CUDA.@profile trace=true raw=true @cuda identity(nothing))

    @test occursin("cuLaunchKernel", str)
    @test occursin("identity()", str)

    @test occursin("ID", str)

    @test occursin("cuCtxSynchronize", str)
end

# benchmarked profile
let
    str = string(CUDA.@bprofile @cuda identity(nothing))
    @test occursin("cuLaunchKernel", str)
    @test occursin("identity()", str)
    @test !occursin("cuCtxGetCurrent", str)

    str = string(CUDA.@bprofile raw=true @cuda identity(nothing))
    @test occursin("cuLaunchKernel", str)
    @test occursin("identity()", str)
    @test occursin("cuCtxGetCurrent", str)
end

if CUPTI.version() != v"13.0.0" # NVIDIA/NVTX#125

# NVTX markers
let
    str = string(CUDA.@profile trace=true NVTX.@mark "a marker")
    @test occursin("NVTX marker", str)
    @test occursin("a marker", str)
end

# with NVTX's GC hooks, collections record NVTX ranges while profiling, which made CUPTI
# call back into Julia from within the collection when it needed a new buffer (#3536)
let
    script = """
        using CUDA
        a = CUDA.zeros(Float32, 16)
        CUDA.@profile nothing
        results = CUDA.@profile for i in 1:200_000
            a .+= 1f0
            i % 50 == 0 && GC.gc(false)
        end
        print(count(isequal("GC"), results.nvtx.name))
        """
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) --color=no -e $script`
    out, err = Pipe(), Pipe()
    proc = run(pipeline(addenv(cmd, "JULIA_NVTX_CALLBACKS" => "gc"); stdout=out, stderr=err);
               wait=false)
    close(out.in)
    close(err.in)
    output, errors = @async(read(out, String)), @async(read(err, String))
    if timedwait(() -> process_exited(proc), 600) != :ok
        kill(proc, Base.SIGKILL)  # the deadlocked process ignores SIGTERM
    end
    wait(proc)
    success(proc) || print(fetch(errors))
    @test success(proc)
    @test parse(Int, fetch(output)) >= 4000
end

# NVTX ranges
let
    str = string(CUDA.@profile trace=true NVTX.@range "a range" identity(nothing))
    @test occursin("NVTX ranges", str)
    @test occursin("a range", str)
end

# unnamed NVTX ranges
let
    results = CUDA.@profile NVTX.range_end(NVTX.range_start())
    @test occursin("NVTX ranges", string(results))
    json = JSON.parse(repr("application/json", results))
    @test only(json["nvtx"])["name"] === nothing
end

# JSON output
let
    @cuda identity(nothing)
    results = CUDA.@profile @cuda identity(nothing)
    json = JSON.parse(repr("application/json", results))
    @test json["view"] == "summary"
    @test json["duration_ns"] > 0
    kernel = only(filter(row -> occursin("identity", row["name"]), json["device"]))
    @test kernel["calls"] == 1
    @test kernel["total_ns"] == kernel["mean_ns"] > 0
    @test kernel["std_ns"] === nothing
    @test any(row -> startswith(row["name"], "cuLaunchKernel"), json["host"])
    @test isempty(json["nvtx"])

    # the trace view matches API calls with the kernels they launched
    results = CUDA.@profile trace=true @cuda threads=2 blocks=3 identity(nothing)
    json = JSON.parse(repr("application/json", results))
    @test json["view"] == "trace"
    kernel = only(filter(row -> occursin("identity", row["name"]), json["device"]))
    @test kernel["block"] == [2, 1, 1]
    @test kernel["grid"] == [3, 1, 1]
    @test kernel["size_bytes"] === nothing
    launch = only(filter(row -> row["id"] == kernel["id"], json["host"]))
    @test startswith(launch["name"], "cuLaunchKernel")
    @test launch["start_ns"] < kernel["start_ns"]

    # names are escaped
    name = "a \"range\"\n\twith \\ characters → \e"
    results = CUDA.@profile trace=true NVTX.@range name identity(nothing)
    json = JSON.parse(repr("application/json", results))
    @test json["schema_version"] == 1
    @test only(json["nvtx"])["name"] == name

    # ranges that don't end have no duration
    results = CUDA.@profile trace=true NVTX.range_start(; message="open")
    json = JSON.parse(repr("application/json", results))
    @test only(json["nvtx"])["duration_ns"] === nothing

    # the raw trace includes everything
    results = CUDA.@profile trace=true raw=true @cuda identity(nothing)
    json = JSON.parse(repr("application/json", results))
    @test json["raw"]
    @test any(row -> row["name"] == "cuCtxSynchronize", json["host"])
end

# timeline export
let
    @cuda identity(nothing)
    results = CUDA.@profile NVTX.@range "outer" begin
        NVTX.@mark "a marker"
        NVTX.@range "inner" @cuda identity(nothing)
        NVTX.range_start(; message="open")
    end
    trace = JSON.parse(sprint(CUDA.Profile.write_trace, results))
    events = trace["traceEvents"]
    named(ph, name) = filter(e -> e["ph"] == ph && occursin(name, e["name"]), events)

    # API calls are connected to the kernels they launched
    launch = only(named("X", "cuLaunchKernel"))
    kernel = only(named("X", "identity"))
    @test kernel["pid"] != launch["pid"]
    @test kernel["ts"] > launch["ts"]
    @test launch["flow_out"] && kernel["flow_in"]
    @test launch["bind_id"] == kernel["bind_id"]
    @test count(e -> haskey(e, "bind_id"), events) == 2

    # NVTX ranges nest on their thread, unless they don't end
    outer, inner = only(named("X", "outer")), only(named("X", "inner"))
    @test outer["tid"] == inner["tid"] == launch["tid"]
    @test outer["ts"] <= inner["ts"] <= launch["ts"]
    @test launch["ts"] + launch["dur"] <= inner["ts"] + inner["dur"] <= outer["ts"] + outer["dur"]
    @test length(named("b", "open")) == length(named("e", "open")) == 1
    @test only(named("i", "a marker"))["tid"] == launch["tid"]

    # the tracks are named
    names = [e["args"]["name"] for e in events if e["ph"] == "M" && e["name"] == "thread_name"]
    @test "Thread 1" in names
    @test any(startswith("Stream"), names)

    # the trace can be written to a file
    mktempdir() do dir
        path = joinpath(dir, "trace.json")
        CUDA.Profile.write_trace(path, results)
        @test JSON.parsefile(path) == trace
    end

    # the raw trace includes everything
    trace = JSON.parse(sprint(io -> CUDA.Profile.write_trace(io, results; raw=true)))
    @test any(e -> e["name"] == "cuCtxSynchronize", trace["traceEvents"])
end

# timeline export of events that complicate nesting and flows
let
    results = synthetic_results(;
        host=[(; id=2, start=2.0, stop=3.0, name="cuLaunchKernel"),
              (; id=3, start=5.0, stop=6.0, name="cuMemcpyHtoDAsync")],
        device=[(; id=2, start=4.0, stop=5.0), (; id=2, start=5.0, stop=6.0)],
        nvtx=[# a marker where an API call starts
              (; id=1, start=2.0, type=:instant, name="marker"),
              # a range that starts within an API call, and ends after it
              (; id=2, start=2.5, type=:start, name="crossing"),
              (; id=2, start=4.0, type=:end),
              # a range that ends within an API call
              (; id=3, start=4.5, type=:start, name="ending"),
              (; id=3, start=5.5, type=:end),
              # ranges that nest with the API calls
              (; id=4, start=1.5, type=:start, name="outer"),
              (; id=4, start=7.0, type=:end),
              (; id=5, start=2.0, type=:start, name="inner"),
              (; id=5, start=2.5, type=:end)])
    events = JSON.parse(sprint(CUDA.Profile.write_trace, results))["traceEvents"]
    phases(name) = sort([e["ph"] for e in events if e["name"] == name])
    @test phases("crossing") == phases("ending") == ["b", "e"]
    @test phases("outer") == phases("inner") == ["X"]
    @test phases("marker") == ["i"]

    # the launch flows out of the API call into both operations
    @test only(e["name"] for e in events if get(e, "flow_out", false)) == "cuLaunchKernel"
    @test count(e -> get(e, "flow_in", false), events) == 2
end

# operations that overlap count once towards the time the GPU was busy
let
    function wait_cycles(cycles)
        t0 = clock(UInt64)
        while clock(UInt64) - t0 < cycles end
        return
    end
    streams = [CuStream() for _ in 1:2]
    launch() = for s in streams
        @cuda stream=s wait_cycles(UInt64(10_000_000))
    end
    launch()
    synchronize()

    results = CUDA.@profile launch()
    json = JSON.parse(repr("application/json", results))
    kernels = filter(row -> occursin("wait_cycles", row["name"]), json["device"])
    @test only(kernels)["calls"] == 2
    device = only(json["devices"])
    @test device["busy_ns"] == json["device_busy_ns"]
    @test device["busy_ns"] + device["idle_ns"] ≈ json["duration_ns"]
    @test only(kernels)["max_ns"] <= device["busy_ns"] < only(kernels)["total_ns"]
    @test occursin(r"GPU was busy for .* and idle for", string(results))

    # intervals that are disjoint, overlap, and nest, from 1 s into the trace
    results = synthetic_results(;
        host=[(; id=2, start=1.5, stop=1.6, name="cuLaunchKernel")],
        device=[(; id=2, start=2.0 + t0, stop=2.0 + t1)
                for (t0, t1) in [(0, 2), (1, 3), (1.5, 2.5), (5, 6), (6, 7)]])
    json = JSON.parse(repr("application/json", results))
    @test json["device_busy_ns"] ≈ 5e9
    @test only(json["devices"])["idle_ns"] ≈ 93e9
    @test json["device_time_ns"] ≈ 7e9
end

end

end
end

############################################################################################

# the range profiler of CUPTI 12.8 to 13.1 doesn't work
broken_range_profiler = can_use_cupti() && v"12.8" <= CUPTI.version() < v"13.2"

# whether this device and driver let us read performance counters
can_collect_metrics = can_use_cupti() && CUPTI.has_range_profiler() &&
                      !broken_range_profiler && try
    @retry_busy CUDA.@metrics metrics=["gpu__time_duration.sum"] nothing
    true
catch err
    msg = sprint(showerror, err)
    contains(msg, "not supported on this device") || contains(msg, "performance counters") ||
        contains(msg, "did not collect hardware metrics") ||
        contains(msg, "did not record when kernels executed") || rethrow()
    @warn "Cannot collect hardware metrics, skipping tests" exception=err
    false
end

@testset "metrics" begin

@test_throws ArgumentError @macroexpand CUDA.@metrics :memory nothing

if can_use_cupti() && !CUPTI.has_range_profiler()
    @test_throws "CUDA 12.6 Update 2" CUDA.@metrics error("must not execute")
elseif broken_range_profiler
    @test_throws "CUDA 13.2" CUDA.@metrics error("must not execute")
end

if can_collect_metrics

function increment(a)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    i <= length(a) && (@inbounds a[i] += 1f0)
    return
end

a = CUDA.zeros(Float32, 1024)

@testset "finalizers" begin
    @test finalizes_within(f -> @retry_busy CUDA.@metrics f())
end

@testset "overview" begin
    p = @retry_busy CUDA.@metrics @cuda threads=32 blocks=32 increment(a)
    @test p isa CUDA.Profile.KernelProfile
    @test length(p.kernels.id) == 1
    @test p.passes >= 1
    @test size(p.values) == (1, length(p.metrics))
    @test p.values[1, 1] > 0    # duration

    str = string(p)
    @test occursin("increment", str)
    @test occursin("Compute", str)
    @test !any(line -> endswith(line, r"[ │┐┤┘]"), split(str, '\n'))
    @test !occursin("bold", str)

    json = JSON.parse(repr("application/json", p))
    @test json["view"] == "summary"
    @test json["launches"] == 1
    @test json["passes"] == p.passes
    @test [metric["name"] for metric in json["metrics"]] == p.metrics
    @test json["metrics"][1]["unit"] == "nanosecond"
    kernel = only(json["kernels"])
    @test occursin("increment", kernel["name"])
    @test kernel["calls"] == 1
    @test kernel["time_fraction"] == 1
    @test kernel["metrics"][p.metrics[1]] == p.values[1, 1]

    p = @retry_busy CUDA.@metrics trace=true @cuda threads=32 blocks=32 increment(a)
    json = JSON.parse(repr("application/json", p))
    @test json["view"] == "trace"
    kernel = only(json["kernels"])
    @test kernel["id"] == only(p.kernels.id)
    @test kernel["grid"] == [32, 1, 1]
    @test kernel["block"] == [32, 1, 1]

    str = string(@retry_busy CUDA.@metrics trace=true @cuda threads=32 blocks=32 increment(a))
    @test occursin("ID", str)
    @test occursin("Blocks", str)

    str = string(@retry_busy CUDA.@metrics metrics=:memory @cuda threads=32 blocks=32 increment(a))
    @test occursin("DRAM traffic", str)

    @test occursin("No kernels", string(@retry_busy CUDA.@metrics nothing))
    json = JSON.parse(repr("application/json", @retry_busy CUDA.@metrics nothing))
    @test json["launches"] == 0
    @test isempty(json["kernels"])
end

@testset "raw metrics" begin
    p = @retry_busy CUDA.@metrics metrics=["smsp__warps_launched.sum", "smsp__average_warp_latency.ratio"] begin
        @cuda threads=32 blocks=32 increment(a)
    end
    @test p.metrics == ["smsp__warps_launched.sum", "smsp__average_warp_latency.ratio"]
    @test p.values[1, 1] == 32
    # ratios aren't percentages
    @test !occursin("%", string(p))

    # without a duration, there's no time fraction, and units aren't known
    json = JSON.parse(repr("application/json", p))
    kernel = only(json["kernels"])
    @test kernel["time_fraction"] === nothing
    @test kernel["metrics"]["smsp__warps_launched.sum"] == 32
    @test all(metric -> metric["unit"] === nothing, json["metrics"])
end

@testset "attribution" begin
    # the number of warps uniquely identifies each launch. on MIG, the counters also include
    # work that other processes run on the same GPU instance (a documented limitation of
    # NVIDIA's profilers), which only ever adds warps to random ranges, so measure again
    # when that happens. an attribution bug would show up in every attempt.
    local p
    for attempt in 1:5
        p = @retry_busy CUDA.@metrics metrics=["smsp__warps_launched.sum"] begin
            @sync for task in 1:4
                Threads.@spawn for i in 1:4
                    @cuda threads=32 blocks=10task+i increment(a)
                end
            end
        end
        p.values[:, 1] == [grid.x for grid in p.kernels.grid] && break
    end
    @test length(p.kernels.id) == 16
    @test p.values[:, 1] == [grid.x for grid in p.kernels.grid]
end

@testset "contexts" begin
    # only kernels in the profiled context are reported, even if the code switches
    ctx = context()
    CuContext(device()) do other
        context!(ctx)
        p = @retry_busy CUDA.@metrics begin
            @cuda increment(a)
            context!(other)
        end
        @test context() == other
        @test length(p.kernels.id) == 1
        context!(ctx)
    end
end

@testset "replay" begin
    # kernels are replayed to collect metrics, which shouldn't affect results
    fill!(a, 0)
    p = @retry_busy CUDA.@metrics @cuda threads=32 blocks=32 increment(a)
    @test p.passes > 1
    @test all(isequal(1), Array(a))
end

@testset "errors" begin
    executed = Ref(false)
    @test_throws "Metric 'foo__bar.sum' is not available" @retry_busy CUDA.@metrics metrics=["foo__bar.sum"] (executed[] = true)
    @test_throws ArgumentError CUDA.@metrics metrics=:unknown (executed[] = true)
    @test !executed[]

    @test_throws "at most" @retry_busy CUDA.@metrics begin
        for i in 1:CUDA.Profile.max_launches+1
            @cuda increment(a)
        end
    end

    stream = CuStream(; flags=CUDA.STREAM_NON_BLOCKING)
    graph = capture(; stream) do
        @cuda stream=stream increment(a)
    end
    exec = instantiate(graph)
    @test_throws "CUDA graphs" @retry_busy CUDA.@metrics begin
        launch(exec, stream)
        synchronize(stream)
    end

    @test_throws "profiled code failed" @retry_busy CUDA.@metrics begin
        @cuda increment(a)
        error("profiled code failed")
    end
    @test length(@retry_busy CUDA.@metrics(@cuda increment(a)).kernels.id) == 1
    @test CUDA.@profile(@cuda increment(a)) isa CUDA.Profile.ProfileResults
end

@testset "nesting" begin
    nested(f) = @retry_busy CUDA.@metrics begin
        @cuda increment(a)
        try
            f()
        catch err
            err
        end
    end
    inner_metrics() = CUDA.@metrics metrics=["gpu__time_duration.sum"] @cuda increment(a)
    inner_trace() = CUDA.@profile @cuda increment(a)

    for inner in (inner_metrics, inner_trace)
        # in the same task
        p = nested(inner)
        @test length(p.kernels.id) == 1

        # in another task, which would deadlock if it waited for the outer session
        p = nested(() -> fetch(Threads.@spawn inner()))
        @test length(p.kernels.id) == 1
    end

    # collecting metrics while tracing activity
    str = string(CUDA.@profile begin
        @cuda increment(a)
        try
            inner_metrics()
        catch
        end
    end)
    @test occursin("increment", str)

    @test length(@retry_busy CUDA.@metrics(@cuda increment(a)).kernels.id) == 1
end

end

end

############################################################################################

@testset "sampling" begin

@test_throws ArgumentError @macroexpand CUDA.@sample :flat nothing
@test_throws ArgumentError CUDA.@sample period=4 error("must not execute")
@test_throws ArgumentError CUDA.@sample format=:graph error("must not execute")

# a cubin with a function `kernel`, whose line table has the given sequences of rows
# (address, line, context row, offset of the inlined function's name), and end addresses.
# ptxas relocates the line table with addends for sm_90 and newer, and without (`rel`) before.
function line_table_cubin(sequences; rel::Bool=false)
    uleb(x) = (out = UInt8[]; while true; b = UInt8(x & 0x7f); x >>= 7;
               x == 0 ? (push!(out, b); return out) : push!(out, b | 0x80); end)
    sleb(x) = (out = UInt8[]; while true; b = UInt8(x & 0x7f); x >>= 7;
               done = (x == 0 && b & 0x40 == 0) || (x == -1 && b & 0x40 != 0);
               done ? (push!(out, b); return out) : push!(out, b | 0x80); end)
    bytes(x) = collect(reinterpret(UInt8, [htol(x)]))

    # a DWARF 2 line program, recording where addresses need a relocation
    program = UInt8[]
    relocations = Int[]
    for (rows, stop) in sequences
        append!(program, [0x00, 0x09, 0x02])
        push!(relocations, length(program))
        append!(program, bytes(UInt64(0)))
        address, line, context = 0, 1, 0
        for (row_address, row_line, row_context, func) in rows
            if row_context != context
                operands = vcat(uleb(row_context), uleb(func))
                append!(program, [0x00; uleb(length(operands) + 1); 0x90; operands])
                context = row_context
            end
            append!(program, [0x02; uleb(row_address - address)])
            append!(program, [0x03; sleb(row_line - line)])
            push!(program, 0x01)
            address, line = row_address, row_line
        end
        append!(program, [0x02; uleb(stop - address)])
        append!(program, [0x00, 0x01, 0x01])
    end
    header = UInt8[1, 1, reinterpret(UInt8, Int8(-5)), 14, 13,
                   0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1,     # standard opcode lengths
                   0x00,                                   # no include directories
                   codeunits("test.jl")..., 0, 0, 0, 0,    # one file
                   0x00]
    prefix = 4 + 2 + 4 + length(header)
    debug_line = vcat(bytes(UInt32(prefix - 4 + length(program))), bytes(UInt16(2)),
                      bytes(UInt32(length(header))), header, program)
    relocations .+= prefix

    reloc_section = rel ? ".rel.debug_line" : ".rela.debug_line"
    shstrtab = "\0.shstrtab\0.strtab\0.symtab\0.text.kernel\0.debug_line\0$reloc_section\0.debug_str\0"
    name(section) = findfirst(section * "\0", shstrtab)[1] - 1
    symtab = vcat(zeros(UInt8, 24), bytes(UInt32(1)), UInt8[0x12, 0], bytes(UInt16(4)),
                  bytes(UInt64(0)), bytes(UInt64(0)))
    reloc = reduce(vcat, [vcat(bytes(UInt64(offset)), bytes(UInt64(1) << 32 | 1),
                               rel ? UInt8[] : bytes(Int64(0))) for offset in relocations])
    sections = [(".shstrtab", 3, codeunits(shstrtab), 0, 0),
                (".strtab", 3, codeunits("\0kernel\0"), 0, 0),
                (".symtab", 2, symtab, 2, 1),
                (".text.kernel", 1, zeros(UInt8, 64), 0, 0),
                (".debug_line", 1, debug_line, 0, 0),
                (reloc_section, rel ? 9 : 4, reloc, 3, 5),
                (".debug_str", 1, codeunits("\0inner;\0"), 0, 0)]

    data = zeros(UInt8, 64)
    offsets = Int[]
    for (_, _, contents, _, _) in sections
        push!(offsets, length(data))
        append!(data, contents)
    end
    shoff = length(data)
    append!(data, zeros(UInt8, 64))     # the null section
    for ((section, type, contents, link, info), offset) in zip(sections, offsets)
        append!(data, bytes(UInt32(name(section))), bytes(UInt32(type)), bytes(UInt64(0)),
                bytes(UInt64(0)), bytes(UInt64(offset)), bytes(UInt64(length(contents))),
                bytes(UInt32(link)), bytes(UInt32(info)), bytes(UInt64(1)), bytes(UInt64(0)))
    end
    data[1:6] = [0x7f, UInt8('E'), UInt8('L'), UInt8('F'), 2, 1]
    data[0x29:0x30] = bytes(UInt64(shoff))
    data[0x3b:0x3c] = bytes(UInt16(64))
    data[0x3d:0x3e] = bytes(UInt16(length(sections) + 1))
    data[0x3f:0x40] = bytes(UInt16(1))
    return data
end

@testset "line tables" begin
    lines(table, pcs) = map(pcs) do pc
        chain = CUDATools.source_chain(table, pc)
        isempty(chain) ? nothing : chain[end].line
    end

    # the sequences of a function are merged, and don't extend past their end
    table = CUDATools.line_tables(line_table_cubin([([(0, 1, 0, 0)], 16),
                                                    ([(32, 2, 0, 0)], 48)]))["kernel"]
    @test lines(table, [0, 8, 15, 16, 31, 32, 47, 48, 64]) ==
          [1, 1, 1, nothing, nothing, 2, 2, nothing, nothing]

    # in any order, and when one starts where another ends
    table = CUDATools.line_tables(line_table_cubin([([(16, 2, 0, 0), (24, 3, 0, 0)], 32),
                                                    ([(0, 1, 0, 0)], 16)]))["kernel"]
    @test lines(table, [0, 15, 16, 24, 31, 32]) == [1, 1, 2, 3, 3, nothing]

    # relocations without addends, as used for sm_87 and older
    table = CUDATools.line_tables(line_table_cubin([([(0, 1, 0, 0)], 16)]; rel=true))["kernel"]
    @test lines(table, [0, 15, 16]) == [1, 1, nothing]

    # inlined frames, outermost first, without repeating frames that are inlined at their
    # own location (like Julia's `macro expansion` frames in unrolled code)
    table = CUDATools.line_tables(line_table_cubin([([(0, 10, 0, 0), (8, 20, 1, 1),
                                                      (16, 20, 2, 1)], 24)]))["kernel"]
    outer = CUDATools.SourceFrame("", "test.jl", 10)
    inner = CUDATools.SourceFrame("inner", "test.jl", 20)
    @test CUDATools.source_chain(table, 0) == [outer]
    @test CUDATools.source_chain(table, 8) == [outer, inner]
    @test CUDATools.source_chain(table, 16) == [outer, inner]
    @test isempty(CUDATools.source_chain(table, 24))
end

# a profile of a function with instructions generated for `chains`, sampled `counts` times
function fixture_profile(chains, counts; ptx=nothing, kwargs...)
    n = length(chains)
    instructions = (; offset=collect(16 .* (0:n-1)), sass=fill("NOP", n), source=chains)
    f = CUDA.Profile.SampledFunction("kernel()", "_Z6kernelv", instructions,
                                     reshape(counts, n, 1), reshape(counts, n, 1), ptx)
    CUDA.Profile.PCProfile(; device="GPU", period=14, total=sum(counts), dropped=0,
                             reasons=["wait"], functions=[f], kwargs...)
end

@testset "noise floor" begin
    hot = [CUDATools.SourceFrame("", "kernel.jl", 1)]
    cold = [CUDATools.SourceFrame("", "kernel.jl", 2)]
    p = fixture_profile([hot, cold], [1000, 1])
    f = only(p.functions)
    for format in (:tree, :flat)
        @test length(CUDA.Profile.report(f, format, 0).rows.count) == 2
        @test length(CUDA.Profile.report(f, format, 2).rows.count) == 1
    end
    str = string(p)
    @test occursin("kernel.jl:1  kernel", str)
    @test !occursin("kernel.jl:2", str)

    # when every location is hidden, say so instead of claiming there were no samples
    str = string(fixture_profile([hot], [1]))
    @test occursin("below the noise floor", str)
    @test !occursin("No samples", str)
    @test occursin("kernel.jl:1  kernel", string(fixture_profile([hot], [1]; noisefloor=0)))
end

@testset "display" begin
    kernel = CUDATools.SourceFrame("", "kernel.jl", 1)
    load = CUDATools.SourceFrame("load", "kernel.jl", 10)
    ptr = CUDATools.SourceFrame("pointer", "pointer.jl", 20)
    add = CUDATools.SourceFrame("+", "float.jl", 30)
    p = fixture_profile([[kernel], [kernel, load], [kernel, load, ptr], [kernel, add]],
                        [100, 300, 400, 200]; noisefloor=0)

    # a tree of locations, with the samples, stalled samples and self samples on every row
    lines = split(string(p), '\n')
    @test lines[1] == "kernel()"
    @test lines[2] == "1000 samples, 1000 stalled (100%): execution 100%"
    @test lines[4] == "Samples  Stalled  Self  File:Line  Function  [Stall reasons]"
    @test lines[6:9] == ["   1000     1000   100  kernel.jl:1  kernel  [wait 100%]",
                         "    700      700   300  ├─ kernel.jl:10  load  [wait 100%]",
                         "    400      400   400  │ └─ pointer.jl:20  pointer  [wait 100%]",
                         "    200      200   200  └─ float.jl:30  +  [wait 100%]"]

    flat = string(fixture_profile([[kernel], [kernel, load], [kernel, load, ptr], [kernel, add]],
                                  [100, 300, 400, 200]; noisefloor=0, format=:flat))
    @test occursin("\n    700      700   300  kernel.jl:10  load  [wait 100%]\n", flat)

    # plain output has no escape sequences and isn't cropped; colored output is
    @test !occursin('\e', string(p))
    long = fixture_profile([[CUDATools.SourceFrame("f", "x"^500, 1)]], [1]; noisefloor=0)
    @test !occursin('…', string(long))
    colored = sprint(show, p; context=:color => true)
    @test occursin("\e[", colored)
    # with the same text, and a legend
    stripped = replace(colored, r"\e\[[0-9;]*m" => "")
    @test startswith(stripped, string(p))
    @test occursin("colored by kind", stripped)
    @test !occursin('\e', sprint(CUDATools.code_sass, p))
    @test occursin("\e[", sprint(CUDATools.code_sass, p; context=:color => true))

    # PTX code is annotated with the samples of the instructions generated for every line,
    # including those of functions that ptxas inlined
    text = ["", ".func  (.param .b32 func_retval0) callee(", ".param .b32 callee_param_0", ")",
            "{", "mul.f32 \t%f1, %f1, %f1;", "ret;", "}", "", ".func other()", "{", "ret;",
            "}", ".visible .entry _Z6kernelv(", ")", "{", "\$L__tmp0:",
            "ld.global.f32 \t%f1, [%rd1];", "add.f32 \t%f2, %f1, %f1;", "{", "mov.b32 \t%r1, 0;",
            "}", "{ mov.b32 \t%r2, 0; }", "ret;", "}"]
    p = fixture_profile([[kernel], [kernel, load], [kernel, load, ptr], [kernel, add], [kernel]],
                        [100, 300, 400, 150, 50]; ptx=(; text, lines=[18, 19, 19, 6, 0]))
    ptx = sprint(CUDATools.code_ptx, p)
    @test occursin("\n       700      700  wait 100% add.f32 \t%f2, %f1, %f1;\n", ptx)
    @test occursin("\n       150      150  wait 100% mul.f32", ptx)
    @test occursin("// 50 samples (5%) are of instructions without a line of PTX code", ptx)
    @test occursin("callee(", ptx) && !occursin("other", ptx)
    @test count("ret;", ptx) == 2   # past the end of blocks in function bodies
    @test findfirst("callee(", ptx)[1] < findfirst(".entry", ptx)[1]
    @test occursin("\n" * " "^31 * "//  kernel.jl:10  load\n", ptx)
    @test !occursin("kernel.jl", sprint((io, p) -> CUDATools.code_ptx(io, p; debuginfo=:none), p))
    # without the labels of debug information, unless asked for
    @test !occursin("\$L__tmp0", ptx)
    @test occursin("\$L__tmp0", sprint((io, p) -> CUDATools.code_ptx(io, p; raw=true), p))
    @test !occursin('\e', ptx)
    @test occursin("\e[", sprint(CUDATools.code_ptx, p; context=:color => true))
    # the JSON contains the PTX code, and the line of every instruction
    json = JSON.parse(repr("application/json", p))
    @test json["ptx"] == [join(text, '\n')]
    @test only(json["functions"])["ptx"] == 0
    @test [inst["ptx_line"] for inst in only(json["functions"])["instructions"]] ==
          [18, 19, 19, 6, nothing]

    # code without line information has no PTX code
    @test occursin("No PTX code available", sprint(CUDATools.code_ptx, fixture_profile([[kernel]], [1])))

    # cropped output fits the display
    cropped = sprint(show, MIME"text/plain"(), p; context=(:limit => true, :displaysize => (24, 30)))
    @test all(line -> textwidth(line) <= 30, split(cropped, '\n')[1:9])

    # stall reasons that newer GPUs may add are grouped by their name
    @test CUDA.Profile.stall_kinds[CUDA.Profile.stall_kind("future_throttle")] == "throttle"
    @test CUDA.Profile.stall_kinds[CUDA.Profile.stall_kind("unknown")] == "other"
end

if can_use_cupti() && !CUPTI.has_pc_sampling()
    @test_throws "CUDA 13" CUDA.@sample error("must not execute")
end

# whether this device and driver let us sample PCs
can_sample = can_use_cupti() && CUPTI.has_pc_sampling() && try
    @retry_busy CUDA.@sample nothing
    true
catch err
    msg = sprint(showerror, err)
    contains(msg, "does not support PC sampling") || contains(msg, "performance counters") ||
        contains(msg, "another process") || contains(msg, "stall reasons") || rethrow()
    @warn "Cannot sample PCs, skipping tests" exception=err
    false
end

if can_sample

# a kernel that spends its time on a known line
spin_line = @__LINE__() + 5
function spin(a, n)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    x = a[i]
    for j in 1:n
        x = sqrt(x + Float32(j))
    end
    a[i] = x
    return
end
a = CUDA.ones(Float32, 256 * 64)
spin_kernel() = @cuda threads=256 blocks=64 spin(a, 20_000)
spin_kernel()
synchronize()

is_spin(f) = startswith(f.name, "spin(")
location(frame) = (basename(frame.file), frame.line)

@testset "finalizers" begin
    @test finalizes_within(f -> @retry_busy CUDA.@sample f())
end

@testset "attribution" begin
    p = @retry_busy CUDA.@sample spin_kernel()
    @test p.total > 0
    f = only(filter(is_spin, p.functions))
    @test sum(f.samples) > 0

    # the kernel spends most of its time on the line in the loop body
    r = CUDA.Profile.report(f, :tree, 0)
    roots = findall(iszero, r.rows.depth)
    hottest = roots[argmax(r.rows.count[roots])]
    @test location(r.rows.path[hottest][end]) == ("profile.jl", spin_line)
    # with `sqrt` inlined into it
    children = [location(path[end]) for (depth, path) in zip(r.rows.depth, r.rows.path)
                if depth == 1 && path[1] == r.rows.path[hottest][1]]
    @test any(((file, _),) -> file == "math.jl", children)

    str = string(p)
    @test occursin("Samples  Stalled  Self  File:Line  Function", str)
    @test occursin(Regex("\\n +\\d+ +\\d+ +\\d+  \\S*profile.jl:$spin_line  spin"), str)
    @test occursin("Total samples: $(p.total)", str)

    flat = string(@retry_busy CUDA.@sample format=:flat spin_kernel())
    @test occursin("Samples  Stalled  Self  File:Line  Function", flat)
    @test occursin("profile.jl:$spin_line  spin", flat)

    # the noise floor hides locations
    rows(noisefloor) = length(CUDA.Profile.report(f, :tree, noisefloor).rows.count)
    @test rows(0) >= rows(2) > 0
    @test rows(1000) == 0

    # frames inlined at their own location (`macro expansion`) aren't repeated in unrolled
    # code, like the loads in a reduction. the reduction is short, so sample more often
    # than the default to be sure to hit it on small devices (e.g. MIG slices).
    x = CUDA.rand(Float32, 2^20)
    sum(x)
    chains = [chain for f in @retry_busy CUDA.@sample(period=10, sum(x)).functions
                    for chain in f.instructions.source]
    @test any(chain -> length(chain) > 1, chains)
    for chain in chains
        @test all(i -> chain[i] != chain[i+1], 1:length(chain)-1)
    end

    sass = sprint(CUDATools.code_sass, p)
    @test occursin("/*0000*/", sass)
    @test occursin("MUFU.RSQ", sass)
    @test occursin("profile.jl:$spin_line  spin", sass)

    @test !occursin("profile.jl:", sprint((io, p) -> CUDATools.code_sass(io, p; debuginfo=:none), p))

    # the PTX code embedded in the cubin maps the samples onto lines of it
    @test f.ptx !== nothing
    @test sum(f.samples[f.ptx.lines .> 0, :]) > 0
    ptx = sprint(CUDATools.code_ptx, p)
    @test occursin(".entry", ptx)
    @test occursin("sqrt", ptx)
    @test occursin("profile.jl:$spin_line  spin", ptx)
    @test occursin(r"\n +\d+ +\d+ .*\w+\.\w+\s", ptx)
end

@testset "JSON" begin
    p = @retry_busy CUDA.@sample spin_kernel()
    json = JSON.parse(repr("application/json", p))
    @test json["view"] == "tree"
    @test json["total_samples"] == p.total
    @test json["period"] == 14
    @test json["reasons"] == p.reasons
    f = only(filter(f -> startswith(f["name"], "spin("), json["functions"]))
    @test f["count"] == sum(only(filter(is_spin, p.functions)).samples)
    @test sum(sum(values(inst["count"])) for inst in f["instructions"]) == f["count"]
    @test all(f["locations"]) do loc
        loc["frame"] === nothing || 0 <= loc["frame"] < length(json["frames"])
    end
    frame(id) = json["frames"][id+1]
    @test any(f["locations"]) do loc
        loc["depth"] == 0 && loc["frame"] !== nothing &&
            frame(loc["frame"])["line"] == spin_line &&
            endswith(frame(loc["frame"])["file"], "profile.jl")
    end
    @test occursin(".entry", json["ptx"][f["ptx"]+1])
    @test any(inst -> inst["ptx_line"] !== nothing, f["instructions"])

    json = JSON.parse(repr("application/json", @retry_busy CUDA.@sample format=:flat spin_kernel()))
    @test json["view"] == "flat"
    @test all(loc -> loc["depth"] == 0, only(filter(f -> startswith(f["name"], "spin("),
                                                     json["functions"]))["locations"])
end

@testset "no kernels" begin
    p = @retry_busy CUDA.@sample nothing
    @test isempty(p.functions)
    @test occursin("No samples", string(p))
    @test isempty(JSON.parse(repr("application/json", p))["functions"])
end

@testset "tasks" begin
    p = @retry_busy CUDA.@sample fetch(Threads.@spawn (spin_kernel(); synchronize()))
    @test any(is_spin, p.functions)
end

@testset "errors" begin
    @test_throws "boom" @retry_busy CUDA.@sample error("boom")
    @test any(is_spin, @retry_busy CUDA.@sample(spin_kernel()).functions)

    @test_throws "another profiling session" @retry_busy CUDA.@sample CUDA.@sample nothing
    # `@metrics` rejects unsupported setups before checking for other sessions
    can_collect_metrics &&
        @test_throws "another profiling session" @retry_busy CUDA.@sample CUDA.@metrics nothing
    @test_throws "sampling PCs" @retry_busy CUDA.@sample CUDA.@profile nothing
    @test_throws "another profiling session" CUDA.@profile CUDA.@sample nothing
end

@testset "SASS reflection" begin
    message(err) = err isa Exception ? sprint(showerror, err) : repr(err)

    # reflection that waits for the session would deadlock when the sampled code waits
    # for it, so it's rejected, in the same task or another one
    sass() = CUDATools.code_sass(() -> nothing, devnull)
    device_sass() = @device_code_sass io=devnull nothing
    for inner in (sass, device_sass), nested in (inner, () -> fetch(@async inner()))
        err = bounded(() -> CUDA.@sample nested())
        @test contains(message(err), "Cannot generate SASS code while")
    end

    # reflection already waiting for CUPTI when a session claims it gives up as well. the
    # session's admission can't be paused, so hold the lock and flag the session manually.
    claimed, release = Channel{Nothing}(1), Channel{Nothing}(1)
    holder = Threads.@spawn lock(CUPTI.callback_lock) do
        put!(claimed, nothing)
        take!(release)
    end
    take!(claimed)
    waiting = Threads.@spawn try
        sass()
    catch err
        err
    end
    sleep(1)
    CUDA.Profile.exclusive_active[] = true
    timedwait(() -> istaskdone(waiting), 120)
    CUDA.Profile.exclusive_active[] = false
    put!(release, nothing)
    wait(holder)
    @test contains(message(fetch(waiting)), "Cannot generate SASS code while")

    # sampling within reflection
    for nested in (() -> CUDA.@sample(nothing), () -> fetch(@async CUDA.@sample(nothing)))
        err = bounded(() -> CUDATools.code_sass(nested, devnull))
        @test contains(message(err), "Cannot sample PCs while CUPTI callbacks are in use")
    end

    @test any(is_spin, @retry_busy CUDA.@sample(spin_kernel()).functions)
    @test CUDATools.code_sass(() -> nothing, devnull) === nothing
end

# CUPTI 13.2+ breaks when loading a module after PC sampling was disabled
@testset "module loading" begin
    p = @retry_busy CUDA.@sample spin_kernel()
    b = CUDA.zeros(Float32, 1)
    p = @retry_busy CUDA.@sample (b .+= 1; synchronize())
    @test p isa CUDA.Profile.PCProfile
    @test CUDA.@profile(b .+= 2) isa CUDA.Profile.ProfileResults
    # whether or not metrics can be collected here, CUPTI shouldn't be broken
    metrics = try
        @retry_busy CUDA.@metrics b .+= 3
    catch err
        err
    end
    @test !occursin("INVALID_OPERATION", sprint(show, metrics))
    @test any(is_spin, @retry_busy CUDA.@sample(spin_kernel()).functions)
end

end

end

end
