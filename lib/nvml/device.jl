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

function name(dev::Device)
    buf = Vector{Cchar}(undef, NVML_DEVICE_NAME_V2_BUFFER_SIZE)
    nvmlDeviceGetName(dev, pointer(buf), length(buf))
    return unsafe_string(pointer(buf))
end

function brand(dev::Device)
    ref = Ref{nvmlBrandType_t}()
    nvmlDeviceGetBrand(dev, ref)
    return ref[]
end

function uuid(dev::Device)
    buf = Vector{Cchar}(undef, NVML_DEVICE_UUID_V2_BUFFER_SIZE)
    nvmlDeviceGetUUID(dev, pointer(buf), length(buf))
    uuid_str = unsafe_string(pointer(buf))
    @assert startswith(uuid_str, "GPU-") || startswith(uuid_str, "MIG-")
    return Base.UUID(uuid_str[5:end])
end

function serial(dev::Device)
    buf = Vector{Cchar}(undef, NVML_DEVICE_SERIAL_BUFFER_SIZE)
    nvmlDeviceGetSerial(dev, pointer(buf), length(buf))
    return unsafe_string(pointer(buf))
end

function index(dev::Device)
    index = Ref{Cuint}()
    nvmlDeviceGetIndex(dev, index)
    return Int(index[])
end


## compute properties

function compute_mode(dev::Device)
    ref = Ref{nvmlComputeMode_t}()
    nvmlDeviceGetComputeMode(dev, ref)
    return ref[]
end

function compute_capability(dev::Device)
    major = Ref{Cint}()
    minor = Ref{Cint}()
    nvmlDeviceGetCudaComputeCapability(dev, major, minor)
    return VersionNumber(major[], minor[])
end

function compute_processes(dev::Device)
    count_ref = Ref{Cuint}(0)
    res = unchecked_nvmlDeviceGetComputeRunningProcesses(dev, count_ref, C_NULL)
    if res == NVML_SUCCESS
        return Dict()
    elseif res !== NVML_ERROR_INSUFFICIENT_SIZE
        throw_api_error(res)
    end

    # "Allocate more space for infos table in case new compute processes are spawned."
    count::Cuint = count_ref[] + 2

    infos = Vector{nvmlProcessInfoV1_t}(undef, count)
    nvmlDeviceGetComputeRunningProcesses(dev, Ref(count), infos)

    return Dict(map(infos[1:count_ref[]]) do info
            pid = Int(info.pid)
            used_gpu_memory = if info.usedGpuMemory == (NVML.NVML_VALUE_NOT_AVAILABLE %
                                                        typeof(info.usedGpuMemory))
                missing
            else
                Int(info.usedGpuMemory)
            end
            pid => (used_gpu_memory=used_gpu_memory,)
        end
    )
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

# default clock speeds
function default_applications_clock(dev::Device)
    info = Dict{Symbol, Int}()
    foreach_clock(dev) do type, clock
        ref = Ref{Cuint}()
        nvmlDeviceGetDefaultApplicationsClock(dev, clock, ref)
        info[type] = Int(ref[])
    end
    return NamedTuple(info)
end

# configured clock speeds
function applications_clock(dev::Device)
    info = Dict{Symbol, Int}()
    foreach_clock(dev) do type, clock
        ref = Ref{Cuint}()
        nvmlDeviceGetApplicationsClock(dev, clock, ref)
        info[type] = Int(ref[])
    end
    return NamedTuple(info)
end

# current clock speeds
function clock_info(dev::Device)
    info = Dict{Symbol, Int}()
    foreach_clock(dev) do type, clock
        ref = Ref{Cuint}()
        nvmlDeviceGetClockInfo(dev, clock, ref)
        info[type] = Int(ref[])
    end
    return NamedTuple(info)
end

# max clock speeds
function max_clock_info(dev::Device)
    info = Dict{Symbol, Int}()
    foreach_clock(dev) do type, clock
        ref = Ref{Cuint}()
        nvmlDeviceGetMaxClockInfo(dev, clock, ref)
        info[type] = Int(ref[])
    end
    return NamedTuple(info)
end

function supported_memory_clocks(dev::Device)
    count_ref = Ref{Cuint}(0)
    unchecked_nvmlDeviceGetSupportedMemoryClocks(dev, count_ref, C_NULL)
    count::Cuint = count_ref[]
    clocks = Vector{Cuint}(undef, count)
    nvmlDeviceGetSupportedMemoryClocks(dev, Ref(count), clocks)
    return Int.(clocks)
end

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

function clock_event_reasons(dev::Device)
    current_events = Ref{Culonglong}()
    if version() >= v"12.2"
        nvmlDeviceGetCurrentClocksEventReasons(dev, current_events)
    else
        nvmlDeviceGetCurrentClocksThrottleReasons(dev, current_events)
    end

    supported_events = Ref{Culonglong}(0)
    if version() >= v"12.2"
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

# watt
function power_usage(dev::Device)
    ref = Ref{Cuint}()
    nvmlDeviceGetPowerUsage(dev, ref)
    return ref[] / 1000
end

# joules
function energy_consumption(dev::Device)
    ref = Ref{Culonglong}()
    nvmlDeviceGetTotalEnergyConsumption(dev, ref)
    return ref[] / 1000
end

# degrees C
function temperature(dev::Device, sensor=NVML_TEMPERATURE_GPU)
    ref = Ref{Cuint}()
    nvmlDeviceGetTemperature(dev, sensor, ref)
    return Int(ref[])
end

# bytes
function memory_info(dev::Device)
    ref = Ref{nvmlMemory_t}()
    nvmlDeviceGetMemoryInfo(dev, ref)
    return (total=Int(ref[].total), free=Int(ref[].free), used=Int(ref[].used))
end

# percent
function utilization_rates(dev::Device)
    ref = Ref{nvmlUtilization_t}()
    nvmlDeviceGetUtilizationRates(dev, ref)
    return (compute=Int(ref[].gpu)/100, memory=Int(ref[].memory)/100)
end
