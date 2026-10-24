import NVTX
using CUDA.Profile: push_row!, filtermask

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
end

# kernel launch (trace)
let
    str = string(CUDA.@profile trace=true @cuda identity(nothing))

    @test occursin("cuLaunchKernel", str)
    @test occursin("identity()", str)

    @test occursin("ID", str)

    @test !occursin("cuCtxSynchronize", str)
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
    CUDA.@metrics metrics=["gpu__time_duration.sum"] nothing
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
    p = CUDA.@metrics @cuda threads=32 blocks=32 increment(a)
    @test p isa CUDA.Profile.KernelProfile
    @test length(p.kernels.id) == 1
    @test p.passes >= 1
    @test size(p.values) == (1, length(p.metrics))
    @test p.values[1, 1] > 0    # duration

    str = string(p)
    @test occursin("increment", str)
    @test occursin("Compute", str)

    str = string(CUDA.@metrics trace=true @cuda threads=32 blocks=32 increment(a))
    @test occursin("ID", str)
    @test occursin("Blocks", str)

    str = string(CUDA.@metrics metrics=:memory @cuda threads=32 blocks=32 increment(a))
    @test occursin("DRAM traffic", str)

    @test occursin("No kernels", string(CUDA.@metrics nothing))
end

@testset "raw metrics" begin
    p = CUDA.@metrics metrics=["smsp__warps_launched.sum", "smsp__average_warp_latency.ratio"] begin
        @cuda threads=32 blocks=32 increment(a)
    end
    @test p.metrics == ["smsp__warps_launched.sum", "smsp__average_warp_latency.ratio"]
    @test p.values[1, 1] == 32
    # ratios aren't percentages
    @test !occursin("%", string(p))
end

@testset "attribution" begin
    # the number of warps uniquely identifies each launch
    p = CUDA.@metrics metrics=["smsp__warps_launched.sum"] begin
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
        p = CUDA.@metrics begin
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
    p = CUDA.@metrics @cuda threads=32 blocks=32 increment(a)
    @test p.passes > 1
    @test all(isequal(1), Array(a))
end

@testset "errors" begin
    executed = Ref(false)
    @test_throws "Metric 'foo__bar.sum' is not available" CUDA.@metrics metrics=["foo__bar.sum"] (executed[] = true)
    @test_throws ArgumentError CUDA.@metrics metrics=:unknown (executed[] = true)
    @test !executed[]

    @test_throws "at most" CUDA.@metrics begin
        for i in 1:CUDA.Profile.max_launches+1
            @cuda increment(a)
        end
    end

    stream = CuStream(; flags=CUDA.STREAM_NON_BLOCKING)
    graph = capture(; stream) do
        @cuda stream=stream increment(a)
    end
    exec = instantiate(graph)
    @test_throws "CUDA graphs" CUDA.@metrics begin
        launch(exec, stream)
        synchronize(stream)
    end

    @test_throws "profiled code failed" CUDA.@metrics begin
        @cuda increment(a)
        error("profiled code failed")
    end
    @test length(CUDA.@metrics(@cuda increment(a)).kernels.id) == 1
    @test CUDA.@profile(@cuda increment(a)) isa CUDA.Profile.ProfileResults
end

@testset "nesting" begin
    nested(f) = CUDA.@metrics begin
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

    @test length(CUDA.@metrics(@cuda increment(a)).kernels.id) == 1
end

end

end

end
