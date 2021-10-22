using CUDA.NVML

macro maybe_unsupported(ex)
    quote
        try
            $(esc(ex))
        catch err
            (isa(err, NVML.NVMLError) && err.code == NVML.ERROR_NOT_SUPPORTED) || rethrow()
        end
    end
end

@testset "system" begin
    @test NVML.version() isa VersionNumber
    @test NVML.driver_version() isa VersionNumber
    @test NVML.cuda_driver_version() == CUDA.version()
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
        @test NVML.uuid(dev) == parent_uuid(cuda_dev)
        NVML.brand(dev)
        @test occursin(NVML.name(dev), name(cuda_dev))
        @maybe_unsupported NVML.serial(dev)

        @maybe_unsupported NVML.power_usage(dev)
        @maybe_unsupported NVML.energy_consumption(dev)

        @maybe_unsupported NVML.utilization_rates(dev)

        NVML.compute_mode(dev)
        @test NVML.compute_capability(dev) == capability(cuda_dev)
    end

    # tests for the compute instance
    let dev = NVML.Device(uuid(cuda_dev); mig)
        @test NVML.uuid(dev) == uuid(cuda_dev)
        @test NVML.name(dev) == name(cuda_dev)

        NVML.memory_info(dev)

        context()
        # FIXME: https://github.com/NVIDIA/gpu-monitoring-tools/issues/63
        #@test getpid() in keys(NVML.compute_processes(dev))
        NVML.compute_processes(dev)
    end
end
