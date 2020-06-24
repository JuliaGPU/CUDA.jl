using CUDA.NVML

@test NVML.version() isa VersionNumber

@test NVML.driver_version() isa VersionNumber
