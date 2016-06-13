# Device type and auxiliary functions

import Base: unsafe_convert

export
    devcount,
    CuDevice, name, totalmem, attribute, capability, architecture,
    list_devices


typealias CuDevice_t Cint

"Return the number of available CUDA devices"
function devcount()
    count_ref = Ref{Cint}()
    @apicall(:cuDeviceGetCount, (Ptr{Cint},), count_ref)
    return count_ref[]
end

immutable CuDevice
    ordinal::Cint
    handle::CuDevice_t

    function CuDevice(i::Integer)
        ordinal = convert(Cint, i)
        handle_ref = Ref{CuDevice_t}()
        @apicall(:cuDeviceGet, (Ptr{CuDevice_t}, Cint), handle_ref, ordinal)
        new(ordinal, handle_ref[])
    end
end

unsafe_convert(::Type{CuDevice_t}, dev::CuDevice) = dev.handle

"Get the name of a CUDA device"
function name(dev::CuDevice)
    const buflen = 256
    buf = Array(Cchar, buflen)
    @apicall(:cuDeviceGetName, (Ptr{Cchar}, Cint, CuDevice_t),
                              buf, buflen, dev.handle)
    buf[end] = 0
    return unsafe_string(pointer(buf))
end

"Get the amount of GPU memory (in bytes) of a CUDA device"
function totalmem(dev::CuDevice)
    mem_ref = Ref{Csize_t}()
    @apicall(:cuDeviceTotalMem, (Ptr{Csize_t}, CuDevice_t), mem_ref, dev.handle)
    return mem_ref[]
end

function attribute(dev::CuDevice, attrcode::Integer)
    value_ref = Ref{Cint}()
    @apicall(:cuDeviceGetAttribute, (Ptr{Cint}, Cint, CuDevice_t),
                                   value_ref, attrcode, dev.handle)
    return value_ref[]
end

"Return the compute capabilities of a CUDA device"
function capability(dev::CuDevice)
    major_ref = Ref{Cint}()
    minor_ref = Ref{Cint}()
    @apicall(:cuDeviceComputeCapability, (Ptr{Cint}, Ptr{Cint}, CuDevice_t),
                                        major_ref, minor_ref, dev.handle)
    return VersionNumber(major_ref[], minor_ref[])
end

"Database of compute capabilities and matching shader model (boolean flag indicates which version of LLVM supports the architecture)"
const architectures = [
    (v"1.0", "sm_10", nothing),
    (v"1.1", "sm_11", nothing),
    (v"1.2", "sm_12", nothing),
    (v"1.3", "sm_13", nothing),
    (v"2.0", "sm_20", v"3.2"),
    (v"2.1", "sm_21", v"3.2"),
    (v"3.0", "sm_30", v"3.2"),
    (v"3.2", "sm_32", v"3.7"),
    (v"3.5", "sm_35", v"3.2"),
    (v"3.7", "sm_37", v"3.7"),
    (v"5.0", "sm_50", v"3.5"),
    (v"5.2", "sm_52", v"3.7"),
    (v"5.3", "sm_53", v"3.7"),
    (v"6.0", "sm_60", nothing),
    (v"6.1", "sm_61", nothing) ]

"Return the most recent supported architecture for a CUDA device"
function architecture(dev::CuDevice)
    cap = capability(dev)
    if cap < architectures[1][1]
        error("No support for SM < $(architectures[1][1])")
    end

    for i = 2:length(architectures)
        if cap < architectures[i][1]
            return architectures[i-1][2]
            break
        end
    end
    return architectures[length(architectures)][2]
end

"List all CUDA devices with their capabilities and attributes"
function list_devices()
    cnt = devcount()
    if cnt == 0
        println("No CUDA-capable device found.")
        return
    end

    for i = 0:cnt-1
        dev = CuDevice(i)
        nam = name(dev)
        tmem = round(Integer, totalmem(dev) / (1024^2))
        cap = capability(dev)

        println("device[$i]: $(nam), capability $(cap.major).$(cap.minor), total mem = $tmem MB")
    end
end
