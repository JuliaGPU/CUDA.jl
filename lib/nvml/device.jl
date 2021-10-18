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



# iteration

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



# queries

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
    res = unsafe_nvmlDeviceGetComputeRunningProcesses(dev, count_ref, C_NULL)
    if res == NVML_SUCCESS
        return nothing
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
