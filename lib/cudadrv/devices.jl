# Device type and auxiliary functions

export
    CuDevice, current_device, has_device,
    name, deviceid, uuid, parent_uuid, totalmem, can_access_peer

"""
    CuDevice(ordinal::Integer)

Get a handle to a compute device.
"""
struct CuDevice
    handle::CUdevice

    function CuDevice(ordinal::Integer)
        device_ref = Ref{CUdevice}()
        cuDeviceGet(device_ref, ordinal)
        new(device_ref[])
    end

    global function current_device()
        device_ref = Ref{CUdevice}()
        res = unchecked_cuCtxGetDevice(device_ref)
        res == ERROR_INVALID_CONTEXT && throw(UndefRefError())
        res != SUCCESS && throw_api_error(res)
        return _CuDevice(device_ref[])
    end

    # for outer constructors
    global _CuDevice(handle::CUdevice) = new(handle)
end

"""
    current_device()

Returns the current device.

!!! warning

    This is a low-level API, returning the current device as known to the CUDA driver.
    For most users, it is recommended to use the [`device`](@ref) method instead.
"""
current_device()

const DEVICE_CPU = _CuDevice(CUdevice(-1))
const DEVICE_INVALID = _CuDevice(CUdevice(-2))

Base.convert(::Type{CUdevice}, dev::CuDevice) = dev.handle

function Base.show(io::IO, ::MIME"text/plain", dev::CuDevice)
  print(io, "CuDevice($(dev.handle)): ")
  if dev == DEVICE_CPU
      print(io, "CPU")
  elseif dev == DEVICE_INVALID
      print(io, "INVALID")
  else
      print(io, "$(name(dev))")
  end
end

"""
    name(dev::CuDevice)

Returns an identifier string for the device.
"""
function name(dev::CuDevice)
    buflen = 256
    buf = Vector{Cchar}(undef, buflen)
    cuDeviceGetName(pointer(buf), buflen, dev)
    buf[end] = 0
    return unsafe_string(pointer(buf))
end

"""
    deviceid(dev::CuDevice)::Int

Get the ID number of the current device of execution. This is a 0-indexed number,
corresponding to the device ID as known to CUDA.
"""
deviceid(dev::CuDevice) = Int(convert(CUdevice, dev))

function uuid(dev::CuDevice)
    driver_version() < v"11.4" && return parent_uuid(dev)

    # returns the MIG UUID if this is a compute instance
    uuid_ref = Ref{CUuuid}()
    cuDeviceGetUuid_v2(uuid_ref, dev)
    Base.UUID(reinterpret(UInt128, reverse([uuid_ref[].bytes...]))[])
end

function parent_uuid(dev::CuDevice)
    uuid_ref = Ref{CUuuid}()
    cuDeviceGetUuid(uuid_ref, dev)
    Base.UUID(reinterpret(UInt128, reverse([uuid_ref[].bytes...]))[])
end

"""
    totalmem(dev::CuDevice)

Returns the total amount of memory (in bytes) on the device.
"""
function totalmem(dev::CuDevice)
    mem_ref = Ref{Csize_t}()
    cuDeviceTotalMem_v2(mem_ref, dev)
    return mem_ref[]
end

function can_access_peer(dev::CuDevice, peer::CuDevice)
    val_ref = Ref{Cint}()
    cuDeviceCanAccessPeer(val_ref, dev, peer)
    return val_ref[] == 1
end


## device iteration

export devices, ndevices

struct DeviceIterator end

"""
    devices()

Get an iterator for the compute devices.
"""
devices() = DeviceIterator()

Base.eltype(::DeviceIterator) = CuDevice

function Base.iterate(iter::DeviceIterator, i=1)
    i >= length(iter) + 1 ? nothing : (CuDevice(i-1), i+1)
end

Base.length(::DeviceIterator) = ndevices()

Base.IteratorSize(::DeviceIterator) = Base.HasLength()

function Base.show(io::IO, ::MIME"text/plain", iter::DeviceIterator)
    print(io, "CUDA.DeviceIterator() for $(length(iter)) devices")
    if !isempty(iter)
        print(io, ":")
        for dev in iter
            print(io, "\n$(deviceid(dev)). $(name(dev))")
        end
    end
end

function ndevices()
    count_ref = Ref{Cint}()
    cuDeviceGetCount(count_ref)
    return count_ref[]
end


## attributes

export attribute, warpsize, capability, memory_pools_supported, unified_addressing

"""
    attribute(dev::CuDevice, code)

Returns information about the device.
"""
function attribute(dev::CuDevice, code::CUdevice_attribute)
    value_ref = Ref{Cint}()
    cuDeviceGetAttribute(value_ref, code, dev)
    return value_ref[]
end

@enum_without_prefix CUdevice_attribute CU_

"""
    warpsize(dev::CuDevice)

Returns the warp size (in threads) of the device.
"""
warpsize(dev::CuDevice) = attribute(dev, DEVICE_ATTRIBUTE_WARP_SIZE)

"""
    capability(dev::CuDevice)

Returns the compute capability of the device.
"""
function capability(dev::CuDevice)
    return VersionNumber(attribute(dev, DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
                         attribute(dev, DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR))
end

memory_pools_supported(dev::CuDevice) =
    CUDA.driver_version() >= v"11.2" &&
    attribute(dev, DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED) == 1
@deprecate has_stream_ordered(dev::CuDevice) memory_pools_supported(dev)

unified_addressing(dev::CuDevice) =
    attribute(dev, DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING) == 1


## p2p attributes

export p2p_attribute

"""
    p2p_attribute(src::CuDevice, dst::CuDevice, code)

Returns information about the P2P relationship between a pair of devices.
"""
function p2p_attribute(src::CuDevice, dst::CuDevice, code::CUdevice_P2PAttribute)
    value_ref = Ref{Cint}()
    cuDeviceGetP2PAttribute(value_ref, code, src, dst)
    return value_ref[]
end

@enum_without_prefix CUdevice_P2PAttribute CU_
