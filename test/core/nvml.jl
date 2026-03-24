using CUDA.NVML

# XXX: can we be more fine grained? or check if we have permissions beforehand?
macro test_maybe(ex)
    quote
        unsupported = try
            $(esc(ex))
            false
        catch err
            isa(err, NVML.NVMLError) &&
            err.code in [NVML.ERROR_NOT_SUPPORTED, NVML.ERROR_NO_PERMISSION]
        end
        @test $(esc(ex)) skip=unsupported
    end
end

if !has_nvml()
@warn "NVML not available, skipping tests"
else

@testset "system" begin
    @test NVML.version() isa VersionNumber
    @test NVML.driver_version() isa VersionNumber
    @test NVML.cuda_driver_version() == CUDA.driver_version()
end

@testset "devices" begin
    let dev = NVML.Device(0)
        @test dev == first(NVML.devices())
        @test NVML.index(dev) == 0

        str = sprint(io->show(io, "text/plain", dev))
        @test occursin("NVML.Device(0)", str)
    end

    cuda_dev = CuDevice(0)
    mig = uuid(cuda_dev) != parent_uuid(cuda_dev)

    # tests for the parent device
    let dev = NVML.Device(parent_uuid(cuda_dev))
        # basic properties
        @test NVML.uuid(dev) == parent_uuid(cuda_dev)
        @test NVML.brand(dev) isa NVML.nvmlBrandType_t
        @test occursin(NVML.name(dev), name(cuda_dev))
        @test_maybe NVML.serial(dev) isa String

        # compute properties
        @test NVML.compute_mode(dev) isa NVML.nvmlComputeMode_t
        @test NVML.compute_capability(dev) == capability(cuda_dev)
        @test_maybe NVML.compute_processes(dev) isa Union{Nothing,Dict}

        # clocks
        @test_maybe NVML.default_applications_clock(dev) isa NamedTuple
        @test_maybe NVML.applications_clock(dev) isa NamedTuple
        @test_maybe NVML.clock_info(dev) isa NamedTuple
        @test_maybe NVML.max_clock_info(dev) isa NamedTuple
        @test_maybe NVML.supported_memory_clocks(dev) isa Vector
        @test_maybe NVML.supported_graphics_clocks(dev) isa Vector
        @test_maybe NVML.clock_event_reasons(dev) isa NamedTuple

        # other queries
        @test_maybe NVML.power_usage(dev) isa Number
        @test_maybe NVML.energy_consumption(dev) isa Number
        @test_maybe NVML.temperature(dev) isa Number
        @test_maybe NVML.memory_info(dev) isa NamedTuple
        @test_maybe NVML.utilization_rates(dev) isa NamedTuple
    end

    # tests for the compute instance
    let dev = NVML.Device(uuid(cuda_dev); mig)
        @test NVML.uuid(dev) == uuid(cuda_dev)
        @test NVML.name(dev) == name(cuda_dev)

        NVML.memory_info(dev)

        context()
        # FIXME: https://github.com/NVIDIA/gpu-monitoring-tools/issues/63
        #@test getpid() in keys(NVML.compute_processes(dev))
        @test_maybe NVML.compute_processes(dev) isa Union{Nothing,Dict}
    end
end

end
