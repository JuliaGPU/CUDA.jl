import JSON
import NVTX
using CUDA.Profile: push_row!, filtermask

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

@testset "profiler" begin

############################################################################################

@testset "helpers" begin

@testset "push_row!" begin
    nt = (id = Int[], name = String[], value = Union{Missing,Int}[])
    push_row!(nt, (id = 1, name = "a"))
    push_row!(nt, (id = 2, name = "b", value = 10))

    @test nt.id == [1, 2]
    @test nt.name == ["a", "b"]
    @test ismissing(nt.value[1])
    @test nt.value[2] == 10
end

@testset "filtermask" begin
    nt = (id = [1, 2, 3, 4, 5], val = [10, 25, 15, 30, 5])
    filtered = filtermask(nt, nt.val .> 15)

    @test filtered.id == [2, 4]
    @test filtered.val == [25, 30]
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
    @test CUPTI.activity_config[] === nothing
    @test CUDA.@profile(identity(nothing)) isa CUDA.Profile.ProfileResults
end

@testset "profiled allocation" begin
    a = nothing
    CUDA.@profile begin
        a = CuArray([1, 2, 3])
    end
    @test Array(a) == [1, 2, 3]
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
        @test CUPTI.activity_config[] === nothing
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

# NVTX ranges
let
    str = string(CUDA.@profile trace=true NVTX.@range "a range" identity(nothing))
    @test occursin("NVTX ranges", str)
    @test occursin("a range", str)
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

end

end
end

############################################################################################

@testset "metrics" begin

@test_throws ArgumentError @macroexpand CUDA.@metrics :memory nothing

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
    # the number of warps uniquely identifies each launch
    p = @retry_busy CUDA.@metrics metrics=["smsp__warps_launched.sum"] begin
        @sync for task in 1:4
            Threads.@spawn for i in 1:4
                @cuda threads=32 blocks=10task+i increment(a)
            end
        end
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
    @test_throws "Metric 'foo__bar.sum' is not available" CUDA.@metrics metrics=["foo__bar.sum"] (executed[] = true)
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

end
