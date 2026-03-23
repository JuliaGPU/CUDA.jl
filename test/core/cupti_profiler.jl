using CUDA

function vadd_kernel(a, b, c)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= length(c)
        @inbounds c[i] = a[i] + b[i]
    end
    return nothing
end

@testset "CUPTI Profiler Host API" begin

@testset "supported_chips" begin
    chips = CUDA.CUPTI.supported_chips()
    @test length(chips) > 0
    @test all(c -> c isa String, chips)
end

@testset "chip_name" begin
    cn = CUDA.CUPTI.chip_name(CUDA.device())
    @test cn isa String
    @test length(cn) > 0
    # chip name should be in the supported list
    @test cn in CUDA.CUPTI.supported_chips()
end

@testset "single_pass_sets" begin
    cn = CUDA.CUPTI.chip_name(CUDA.device())
    sets = CUDA.CUPTI.single_pass_sets(cn)
    @test sets isa Vector{String}
    @test length(sets) > 0
end

@testset "ProfilerHostContext and metric enumeration" begin
    cn = CUDA.CUPTI.chip_name(CUDA.device())
    ctx = CUDA.CUPTI.ProfilerHostContext(cn;
        profiler_type=CUDA.CUPTI.CUPTI_PROFILER_TYPE_PM_SAMPLING)
    try
        # counter metrics
        counters = CUDA.CUPTI.base_metrics(ctx, CUDA.CUPTI.CUPTI_METRIC_TYPE_COUNTER)
        @test length(counters) > 0

        # sub-metrics
        subs = CUDA.CUPTI.sub_metrics(ctx, counters[1], CUDA.CUPTI.CUPTI_METRIC_TYPE_COUNTER)
        @test length(subs) > 0

        # metric properties
        props = CUDA.CUPTI.metric_properties(ctx, counters[1])
        @test props isa CUDA.CUPTI.MetricProperties
        @test props.hw_unit isa String
    finally
        close(ctx)
    end
end

@testset "list_metrics" begin
    metrics = CUDA.CUPTI.list_metrics()
    @test length(metrics) > 0
    m = metrics[1]
    @test haskey(m, :name)
    @test haskey(m, :description)
    @test haskey(m, :hw_unit)
end

@testset "check_profiling_permissions" begin
    result = CUDA.CUPTI.check_profiling_permissions()
    @test result isa Bool
end

@testset "range_profile" begin
    N = 1024 * 1024
    a = CUDA.rand(Float32, N)
    b = CUDA.rand(Float32, N)
    c = CUDA.zeros(Float32, N)

    result = CUDA.CUPTI.range_profile(["sm__cycles_active.avg"]) do
        @cuda threads=256 blocks=cld(N, 256) vadd_kernel(a, b, c)
        CUDA.synchronize()
    end

    @test result isa CUDA.CUPTI.RangeProfileResult
    @test length(result.range_names) >= 1
    @test result.metric_names == ["sm__cycles_active.avg"]
    @test size(result.values, 2) == 1
    # SM cycles should be a positive number
    @test result.values[1, 1] > 0
end

@testset "pm_sample" begin
    N = 1024 * 1024
    a = CUDA.rand(Float32, N)
    b = CUDA.rand(Float32, N)
    c = CUDA.zeros(Float32, N)

    result = CUDA.CUPTI.pm_sample(["sm__cycles_active.avg"];
                                   sampling_interval=UInt64(5000)) do
        for _ in 1:50
            @cuda threads=256 blocks=cld(N, 256) vadd_kernel(a, b, c)
        end
        CUDA.synchronize()
    end

    @test result isa CUDA.CUPTI.PmSamplingResult
    @test result.metric_names == ["sm__cycles_active.avg"]
    @test length(result.samples) > 0
    # at least some samples should have non-zero timestamps
    @test any(s -> s.end_timestamp > s.start_timestamp, result.samples)
end

end
