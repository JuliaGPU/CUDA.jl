# Device type and auxiliary functions

export
    CuDevice, name, totalmem, attribute


"""
    CuDevice(i::Integer)

Get a handle to a compute device.
"""
struct CuDevice
    handle::CUdevice

    # CuDevice is just an integer, but we need (?) to call cuDeviceGet to make sure this
    # integer is valid. to avoid ambiguity, add a bogus argument (cfr. `checkbounds`)
    CuDevice(::Type{Bool}, handle::CUdevice) = new(handle)
end

const DEVICE_CPU = CuDevice(Bool, CUdevice(-1))
const DEVICE_INVALID = CuDevice(Bool, CUdevice(-2))

function CuDevice(ordinal::Integer)
    device_ref = Ref{CUdevice}()
    cuDeviceGet(device_ref, ordinal)
    CuDevice(Bool, device_ref[])
end

Base.convert(::Type{CUdevice}, dev::CuDevice) = dev.handle

Base.:(==)(a::CuDevice, b::CuDevice) = a.handle == b.handle
Base.hash(dev::CuDevice, h::UInt) = hash(dev.handle, h)

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
    totalmem(dev::CuDevice)

Returns the total amount of memory (in bytes) on the device.
"""
function totalmem(dev::CuDevice)
    mem_ref = Ref{Csize_t}()
    cuDeviceTotalMem(mem_ref, dev)
    return mem_ref[]
end


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


## device iteration

export devices

struct DeviceSet end

"""
    devices()

Get an iterator for the compute devices.
"""
devices() = DeviceSet()

Base.eltype(::DeviceSet) = CuDevice

function Base.iterate(iter::DeviceSet, i=1)
    i >= length(iter) + 1 ? nothing : (CuDevice(i-1), i+1)
end

function Base.length(::DeviceSet)
    count_ref = Ref{Cint}()
    cuDeviceGetCount(count_ref)
    return count_ref[]
end

Base.IteratorSize(::DeviceSet) = Base.HasLength()


## convenience attribute getters

export warpsize, capability

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
