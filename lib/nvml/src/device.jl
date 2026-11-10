@public Device, devices, name, brand, uuid, serial, index
@public compute_mode, compute_capability, compute_processes
@public default_applications_clock, applications_clock, clock_info, max_clock_info,
        supported_memory_clocks, supported_graphics_clocks, clock_event_reasons
@public power_usage, energy_consumption, temperature, memory_info, utilization_rates

"""
    NVML.Device(index::Integer) -> NVML.Device
    NVML.Device(uuid::Base.UUID; mig=false) -> NVML.Device
    NVML.Device(serial::String) -> NVML.Device

A GPU as seen by NVML, looked up by its NVML index (starting at 0), its UUID, or its board
serial number. Wraps `nvmlDeviceGetHandleByIndex`, `nvmlDeviceGetHandleByUUID` and
`nvmlDeviceGetHandleBySerial`.

NVML numbers the GPUs in its own order, which ignores `CUDA_VISIBLE_DEVICES` and can
differ from CUDA's, so an NVML index need not match the ordinal of a `CuDevice`. To get the NVML device for a `CuDevice`, look it
up by UUID instead; for a MIG compute instance, set `mig = true`:

```julia
dev = device()
mig = uuid(dev) != parent_uuid(dev)
nvml_dev = NVML.Device(uuid(dev); mig)
```

`NVML.Device(parent_uuid(dev))` gives the physical GPU a MIG instance belongs to.
"""
struct Device
    handle::nvmlDevice_t

    function Device(index::Integer)
        handle_ref = Ref{nvmlDevice_t}()
        nvmlDeviceGetHandleByIndex_v2(index, handle_ref)
        return new(handle_ref[])
    end

    function Device(uuid::Base.UUID; mig::Bool=false)
        handle_ref = Ref{nvmlDevice_t}()
        # XXX: keep the UUID as a string with prefix (GPU or MIG)?
        nvmlDeviceGetHandleByUUID("$(mig ? "MIG" : "GPU")-$uuid", handle_ref)
        return new(handle_ref[])
    end

    function Device(serial::String)
        handle_ref = Ref{nvmlDevice_t}()
        nvmlDeviceGetHandleBySerial(serial, handle_ref)
        return new(handle_ref[])
    end
end

Base.unsafe_convert(::Type{nvmlDevice_t}, dev::Device) = dev.handle

function Base.show(io::IO, ::MIME"text/plain", dev::Device)
    print(io, "NVML.Device($(index(dev))): ")
    print(io, "$(name(dev))")
end


## iteration

struct DeviceIterator end

"""
    NVML.devices() -> NVML.DeviceIterator

An iterator over all GPUs that NVML knows of, as [`NVML.Device`](@ref)s in NVML's order.
Unlike `CUDACore.devices()`, it includes GPUs hidden by `CUDA_VISIBLE_DEVICES`. It is not
exported, so that it doesn't clash with `devices()`.
"""
devices() = DeviceIterator()

Base.eltype(::DeviceIterator) = Device

function Base.iterate(iter::DeviceIterator, i=1)
    i >= length(iter) + 1 ? nothing : (Device(i-1), i+1)
end

function Base.length(::DeviceIterator)
    count_ref = Ref{Cuint}()
    nvmlDeviceGetCount_v2(count_ref)
    return count_ref[]
end

Base.IteratorSize(::DeviceIterator) = Base.HasLength()


## properties

"""
    NVML.name(dev::NVML.Device) -> String

The product name of the device, e.g. `"NVIDIA GeForce RTX 5080"`, or of the MIG compute
instance. Wraps `nvmlDeviceGetName`.

This is a method of `CUDACore.name`, so `name(dev)` works without the `NVML.` prefix.
"""
function name(dev::Device)
    buf = Vector{Cchar}(undef, NVML_DEVICE_NAME_V2_BUFFER_SIZE)
    nvmlDeviceGetName(dev, pointer(buf), length(buf))
    return GC.@preserve buf unsafe_string(pointer(buf))
end

"""
    NVML.brand(dev::NVML.Device) -> nvmlBrandType_t

The brand of the device, e.g. `NVML_BRAND_GEFORCE`. Wraps `nvmlDeviceGetBrand`.
"""
function brand(dev::Device)
    ref = Ref{nvmlBrandType_t}()
    nvmlDeviceGetBrand(dev, ref)
    return ref[]
end

"""
    NVML.uuid(dev::NVML.Device) -> Base.UUID

The UUID of the device, or of the MIG compute instance. It matches `uuid(::CuDevice)` for
the corresponding CUDA device. Wraps `nvmlDeviceGetUUID`.

This is a method of `CUDACore.uuid`, so `uuid(dev)` works without the `NVML.` prefix.
"""
function uuid(dev::Device)
    buf = Vector{Cchar}(undef, NVML_DEVICE_UUID_V2_BUFFER_SIZE)
    nvmlDeviceGetUUID(dev, pointer(buf), length(buf))
    uuid_str = GC.@preserve buf unsafe_string(pointer(buf))
    @assert startswith(uuid_str, "GPU-") || startswith(uuid_str, "MIG-")
    return Base.UUID(uuid_str[5:end])
end

"""
    NVML.serial(dev::NVML.Device) -> String

The serial number printed on the board. Wraps `nvmlDeviceGetSerial`, which only supports
data-center and workstation GPUs: others throw an [`NVMLError`](@ref) with code
`ERROR_NOT_SUPPORTED`.
"""
function serial(dev::Device)
    buf = Vector{Cchar}(undef, NVML_DEVICE_SERIAL_BUFFER_SIZE)
    nvmlDeviceGetSerial(dev, pointer(buf), length(buf))
    return GC.@preserve buf unsafe_string(pointer(buf))
end

"""
    NVML.index(dev::NVML.Device) -> Int

The NVML index of the device, as passed to `NVML.Device(index)`. For a MIG compute
instance, it is the index of the instance within its GPU instead. Wraps
`nvmlDeviceGetIndex`.
"""
function index(dev::Device)
    index = Ref{Cuint}()
    nvmlDeviceGetIndex(dev, index)
    return Int(index[])
end


## compute properties

"""
    NVML.compute_mode(dev::NVML.Device) -> nvmlComputeMode_t

The compute mode of the device, which determines whether several processes can use it at the
same time, e.g. `NVML_COMPUTEMODE_DEFAULT` or `NVML_COMPUTEMODE_EXCLUSIVE_PROCESS`. Wraps
`nvmlDeviceGetComputeMode`.
"""
function compute_mode(dev::Device)
    ref = Ref{nvmlComputeMode_t}()
    nvmlDeviceGetComputeMode(dev, ref)
    return ref[]
end

"""
    NVML.compute_capability(dev::NVML.Device) -> VersionNumber

The compute capability of the device, e.g. `v"12.0"`, as `capability(::CuDevice)`
returns it. Wraps `nvmlDeviceGetCudaComputeCapability`.
"""
function compute_capability(dev::Device)
    major = Ref{Cint}()
    minor = Ref{Cint}()
    nvmlDeviceGetCudaComputeCapability(dev, major, minor)
    return VersionNumber(major[], minor[])
end

const ProcessInfo = @NamedTuple{used_gpu_memory::Union{Int,Missing}}

"""
    NVML.compute_processes(dev::NVML.Device)
        -> Dict{Int,@NamedTuple{used_gpu_memory::Union{Int,Missing}}}

The processes that have a compute context on the device, by process ID, with the GPU memory
in bytes each one uses. The memory use is `missing` when NVML can't report it, e.g. on
Windows with the WDDM driver model. Processes in other containers may be left out. Wraps
`nvmlDeviceGetComputeRunningProcesses`.
"""
function compute_processes(dev::Device)
    processes = Dict{Int,ProcessInfo}()

    count_ref = Ref{Cuint}(0)
    res = unchecked_nvmlDeviceGetComputeRunningProcesses(dev, count_ref, C_NULL)
    if res == NVML_SUCCESS
        return processes
    elseif res !== NVML_ERROR_INSUFFICIENT_SIZE
        throw_api_error(res)
    end

    # "Allocate more space for infos table in case new compute processes are spawned."
    count_ref[] += 2
    infos = Vector{nvmlProcessInfo_v1_t}(undef, count_ref[])
    nvmlDeviceGetComputeRunningProcesses(dev, count_ref, infos)

    for info in view(infos, 1:count_ref[])
        used_gpu_memory = if info.usedGpuMemory == (NVML_VALUE_NOT_AVAILABLE %
                                                    typeof(info.usedGpuMemory))
            missing
        else
            Int(info.usedGpuMemory)
        end
        processes[Int(info.pid)] = (; used_gpu_memory)
    end
    return processes
end


## clocks

function foreach_clock(f, dev::Device)
    for (type, clock) in [:graphics => NVML_CLOCK_GRAPHICS,
                          :sm => NVML_CLOCK_SM,
                          :memory => NVML_CLOCK_MEM,
                          :video => NVML_CLOCK_VIDEO,]
        try
            f(type, clock)
        catch err
            if isa(err, NVML.NVMLError) && err.code == NVML.ERROR_NOT_SUPPORTED
                continue
            end
            rethrow()
        end
    end
end

"""
    NVML.default_applications_clock(dev::NVML.Device) -> NamedTuple

The default applications clocks of the device: the clocks it runs kernels at unless
configured otherwise, in MHz, as a named tuple with entries for the `graphics`, `sm`, `memory` and
`video` clocks. Clocks that the device doesn't report are left out. Wraps
`nvmlDeviceGetDefaultApplicationsClock`.
"""
function default_applications_clock(dev::Device)
    info = Dict{Symbol, Int}()
    foreach_clock(dev) do type, clock
        ref = Ref{Cuint}()
        nvmlDeviceGetDefaultApplicationsClock(dev, clock, ref)
        info[type] = Int(ref[])
    end
    return NamedTuple(info)
end

"""
    NVML.applications_clock(dev::NVML.Device) -> NamedTuple

The applications clocks the device is configured to run kernels at, in MHz, as a named tuple with entries for the `graphics`, `sm`, `memory` and
`video` clocks. Clocks that the device doesn't report are left out. Wraps
`nvmlDeviceGetApplicationsClock`.
"""
function applications_clock(dev::Device)
    info = Dict{Symbol, Int}()
    foreach_clock(dev) do type, clock
        ref = Ref{Cuint}()
        nvmlDeviceGetApplicationsClock(dev, clock, ref)
        info[type] = Int(ref[])
    end
    return NamedTuple(info)
end

"""
    NVML.clock_info(dev::NVML.Device) -> NamedTuple

The current clocks of the device, in MHz, as a named tuple with entries for the `graphics`, `sm`, `memory` and
`video` clocks. Clocks that the device doesn't report are left out. Wraps
`nvmlDeviceGetClockInfo`.
"""
function clock_info(dev::Device)
    info = Dict{Symbol, Int}()
    foreach_clock(dev) do type, clock
        ref = Ref{Cuint}()
        nvmlDeviceGetClockInfo(dev, clock, ref)
        info[type] = Int(ref[])
    end
    return NamedTuple(info)
end

"""
    NVML.max_clock_info(dev::NVML.Device) -> NamedTuple

The maximum clocks of the device, in MHz, as a named tuple with entries for the `graphics`, `sm`, `memory` and
`video` clocks. Clocks that the device doesn't report are left out. Wraps
`nvmlDeviceGetMaxClockInfo`.
"""
function max_clock_info(dev::Device)
    info = Dict{Symbol, Int}()
    foreach_clock(dev) do type, clock
        ref = Ref{Cuint}()
        nvmlDeviceGetMaxClockInfo(dev, clock, ref)
        info[type] = Int(ref[])
    end
    return NamedTuple(info)
end

"""
    NVML.supported_memory_clocks(dev::NVML.Device) -> Vector{Int}

The memory clocks, in MHz, that the applications clocks of the device can be set to. Wraps
`nvmlDeviceGetSupportedMemoryClocks`.
"""
function supported_memory_clocks(dev::Device)
    count_ref = Ref{Cuint}(0)
    unchecked_nvmlDeviceGetSupportedMemoryClocks(dev, count_ref, C_NULL)
    count::Cuint = count_ref[]
    clocks = Vector{Cuint}(undef, count)
    nvmlDeviceGetSupportedMemoryClocks(dev, Ref(count), clocks)
    return Int.(clocks)
end

"""
    NVML.supported_graphics_clocks(dev::NVML.Device, [memory_clock]) -> Vector{Int}

The graphics clocks, in MHz, that the applications clocks of the device can be set to
together with the memory clock `memory_clock` (in MHz), or with any of the
[`supported_memory_clocks`](@ref) if it is omitted. Wraps
`nvmlDeviceGetSupportedGraphicsClocks`.
"""
function supported_graphics_clocks(dev::Device, memory_clock)
    count_ref = Ref{Cuint}(0)
    unchecked_nvmlDeviceGetSupportedGraphicsClocks(dev, memory_clock, count_ref, C_NULL)
    count::Cuint = count_ref[]
    clocks = Vector{Cuint}(undef, count)
    nvmlDeviceGetSupportedGraphicsClocks(dev, memory_clock, Ref(count), clocks)
    return Int.(clocks)
end

supported_graphics_clocks(dev::Device) =
    union(supported_graphics_clocks.(Ref(dev), supported_memory_clocks(dev))...)

"""
    NVML.clock_event_reasons(dev::NVML.Device) -> NamedTuple

Why the clocks of the device are currently reduced, as a named tuple of `Bool`s with
entries for the reasons the device supports:

- `idle`: nothing is running on the GPU;
- `application_setting`: the applications clocks limit the clocks;
- `sw_power_cap`: the driver keeps the power draw below the power limit;
- `hw_slow`: hardware slowdown, because of temperature, power or an external signal;
- `sync_boost`: the GPU is part of a sync boost group;
- `sw_thermal`: the driver keeps the temperature below its limit;
- `hw_thermal`: hardware slowdown because of temperature;
- `hw_power_brake`: hardware slowdown because of an external power brake signal;
- `display_setting`: the display clock settings limit the clocks.

Wraps `nvmlDeviceGetCurrentClocksEventReasons` and
`nvmlDeviceGetSupportedClocksEventReasons`, or their predecessors
`nvmlDeviceGet*ClocksThrottleReasons` on drivers older than R535.
"""
function clock_event_reasons(dev::Device)
    # the ClocksEventReasons API replaced ClocksThrottleReasons in the R535 driver
    has_event_api = driver_version() >= v"535"

    current_events = Ref{Culonglong}()
    if has_event_api
        nvmlDeviceGetCurrentClocksEventReasons(dev, current_events)
    else
        nvmlDeviceGetCurrentClocksThrottleReasons(dev, current_events)
    end

    supported_events = Ref{Culonglong}(0)
    if has_event_api
        nvmlDeviceGetSupportedClocksEventReasons(dev, supported_events)
    else
        nvmlDeviceGetSupportedClocksThrottleReasons(dev, supported_events)
    end

    reasons = [
        # Nothing is running on the GPU and the clocks are dropping to Idle state
        :idle                   => nvmlClocksEventReasonGpuIdle,
        # Clocks have been limited by applications clocks
        :application_setting    => nvmlClocksEventReasonApplicationsClocksSetting,
        # The clocks have been optimized to ensure not to exceed currently set power limits
        :sw_power_cap           => nvmlClocksEventReasonSwPowerCap,
        # Hardware-driven reduction because of thermals, power limits, ...
        :hw_slow                => nvmlClocksThrottleReasonHwSlowdown,
        # This GPU has been added to a Sync boost group with nvidia-smi or DCGM
        :sync_boost             => nvmlClocksEventReasonSyncBoost,
        # Software-driven clock reduction for thermal reasons
        :sw_thermal             => nvmlClocksEventReasonSwThermalSlowdown,
        # Hardware-driven clock reduction for thermal reasons
        :hw_thermal             => nvmlClocksThrottleReasonHwThermalSlowdown,
        # Hardware-driven clock reduction for external (e.g. PSU) power reasons
        :hw_power_brake         => nvmlClocksThrottleReasonHwPowerBrakeSlowdown,
        # GPU clocks are limited by current setting of Display clocks
        :display_setting        => nvmlClocksEventReasonDisplayClockSetting,
    ]
    info = Dict{Symbol, Bool}()
    for (name, reason) in reasons
        if (supported_events[] & reason) == reason
            info[name] = (current_events[] & reason) == reason
        end
    end
    return NamedTuple(info)
end


## other queries

"""
    NVML.power_usage(dev::NVML.Device) -> Float64

The power draw of the device, in watts. On Ampere (except GA100) and newer GPUs, this is
the average over the last second, on older ones the instantaneous power draw. Wraps
`nvmlDeviceGetPowerUsage`.
"""
function power_usage(dev::Device)
    ref = Ref{Cuint}()
    nvmlDeviceGetPowerUsage(dev, ref)
    return ref[] / 1000
end

"""
    NVML.energy_consumption(dev::NVML.Device) -> Float64

The energy the device has consumed since the driver was last loaded, in joules. Take the
difference of two calls to measure the energy used in between. Wraps
`nvmlDeviceGetTotalEnergyConsumption`, which needs a Volta or newer GPU.
"""
function energy_consumption(dev::Device)
    ref = Ref{Culonglong}()
    nvmlDeviceGetTotalEnergyConsumption(dev, ref)
    return ref[] / 1000
end

"""
    NVML.temperature(dev::NVML.Device, sensor=NVML_TEMPERATURE_GPU) -> Int

The temperature of the device, in degrees Celsius, as measured by `sensor`, an
`nvmlTemperatureSensors_t`. Wraps `nvmlDeviceGetTemperature`.
"""
function temperature(dev::Device, sensor=NVML_TEMPERATURE_GPU)
    ref = Ref{Cuint}()
    nvmlDeviceGetTemperature(dev, sensor, ref)
    return Int(ref[])
end

"""
    NVML.memory_info(dev::NVML.Device) -> @NamedTuple{total::Int, free::Int, used::Int}

The total, free and used memory of the device, in bytes, counting the memory of all
processes. For a MIG compute instance, these are the numbers of the instance; for a GPU in
MIG mode, querying the GPU as a whole needs administrator rights. Wraps
`nvmlDeviceGetMemoryInfo`.
"""
function memory_info(dev::Device)
    ref = Ref{nvmlMemory_t}()
    nvmlDeviceGetMemoryInfo(dev, ref)
    return (total=Int(ref[].total), free=Int(ref[].free), used=Int(ref[].used))
end

"""
    NVML.utilization_rates(dev::NVML.Device) -> @NamedTuple{compute::Float64, memory::Float64}

How busy the device was over the driver's last sample period (between 1/6 s and 1 s), as
fractions between 0 and 1: `compute` is the fraction of time a kernel was running, and
`memory` the fraction of time device memory was being read or written. Wraps
`nvmlDeviceGetUtilizationRates`, which doesn't support GPUs in MIG mode.
"""
function utilization_rates(dev::Device)
    ref = Ref{nvmlUtilization_t}()
    nvmlDeviceGetUtilizationRates(dev, ref)
    return (compute=Int(ref[].gpu)/100, memory=Int(ref[].memory)/100)
end
