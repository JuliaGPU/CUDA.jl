# Device type and auxiliary functions

export
    CuDevice, name, totalmem, attribute


const CuDevice_t = Cint

"""
    CuDevice(i::Integer)

Get a handle to a compute device.
"""
struct CuDevice
    handle::CuDevice_t

    # CuDevice is just an integer, but we need (?) to call cuDeviceGet to make sure this
    # integer is valid. to avoid ambiguity, add a bogus argument (cfr. `checkbounds`)
    CuDevice(::Type{Bool}, handle::CuDevice_t) = new(handle)
end

const DEVICE_CPU = CuDevice(Bool, CuDevice_t(-1))
const DEVICE_INVALID = CuDevice(Bool, CuDevice_t(-2))

function CuDevice(ordinal::Integer)
    device_ref = Ref{CuDevice_t}()
    @apicall(:cuDeviceGet, (Ptr{CuDevice_t}, Cint), device_ref, ordinal)
    CuDevice(Bool, device_ref[])
end

Base.convert(::Type{CuDevice_t}, dev::CuDevice) = dev.handle

Base.:(==)(a::CuDevice, b::CuDevice) = a.handle == b.handle
Base.hash(dev::CuDevice, h::UInt) = hash(dev.handle, h)

function Base.show(io::IO, dev::CuDevice)
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
    @apicall(:cuDeviceGetName, (Ptr{Cchar}, Cint, CuDevice_t),
                               buf, buflen, dev)
    buf[end] = 0
    return unsafe_string(pointer(buf))
end

"""
    totalmem(dev::CuDevice)

Returns the total amount of memory (in bytes) on the device.
"""
function totalmem(dev::CuDevice)
    mem_ref = Ref{Csize_t}()
    @apicall(:cuDeviceTotalMem, (Ptr{Csize_t}, CuDevice_t), mem_ref, dev)
    return mem_ref[]
end


"""
    attribute(dev::CuDevice, code)

Returns information about the device.
"""
function attribute(dev::CuDevice, code::CUdevice_attribute)
    value_ref = Ref{Cint}()
    @apicall(:cuDeviceGetAttribute, (Ptr{Cint}, Cint, CuDevice_t),
                                    value_ref, code, dev)
    return value_ref[]
end


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
    @apicall(:cuDeviceGetCount, (Ptr{Cint},), count_ref)
    return count_ref[]
end

Base.IteratorSize(::DeviceSet) = Base.HasLength()


## convenience attribute getters

export warpsize, capability

"""
    warpsize(dev::CuDevice)

Returns the warp size (in threads) of the device.
"""
warpsize(dev::CuDevice) = attribute(dev, WARP_SIZE)

"""
    capability(dev::CuDevice)

Returns the compute capability of the device.
"""
function capability(dev::CuDevice)
    return VersionNumber(attribute(dev, COMPUTE_CAPABILITY_MAJOR),
                         attribute(dev, COMPUTE_CAPABILITY_MINOR))
end
