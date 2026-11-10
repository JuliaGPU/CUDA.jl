# NVML

```@meta
CurrentModule = NVML
```

NVML, the NVIDIA Management Library, reports the state of the GPUs in the system: their
clocks, power draw, temperature, memory use and the processes using them. It ships with the
driver and doesn't need a CUDA context, so it can also see GPUs that CUDA.jl doesn't use.
`CUDA.versioninfo()` uses it when it is available.

The module offers NVML at two levels:

- The functions documented below query a device or the system, and return Julia values,
  in bytes, MHz, watts, joules and degrees Celsius, and fractions for utilization. They are not exported, apart from [`has_nvml`](@ref), so call them as
  `NVML.memory_info(dev)`.
- The generated bindings in `libnvml.jl` follow NVIDIA's C API, and change with the driver
  headers. An `NVML.Device` converts to the `nvmlDevice_t` they take.

Check [`has_nvml`](@ref) before using NVML: CUDA.jl doesn't use it on Tegra devices, and
on some systems the library is missing or fails to initialize. Not every query is supported by
every device, and some need administrator rights: a failing query throws an
[`NVMLError`](@ref), whose `code` tells these cases apart.

NVML numbers devices differently from CUDA, so look up the NVML device for a `CuDevice` by
its UUID:

```julia
dev = device()
nvml_dev = NVML.Device(uuid(dev); mig = uuid(dev) != parent_uuid(dev))
NVML.memory_info(nvml_dev)
```

## Public

```@autodocs
Modules = [NVML]
Private = false
```

## Private

```@autodocs
Modules = [NVML]
Public = false
```
