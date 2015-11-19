# Device type and auxiliary functions

export
    devcount,
    CuDevice, name, totalmem, attribute, capability,
    list_devices


function devcount()
    count_ref = Ref{Cint}()
    @cucall(:cuDeviceGetCount, (Ptr{Cint},), count_ref)
    return count_ref[]
end

immutable CuDevice
    ordinal::Cint
    handle::Cint

    function CuDevice(i::Integer)
        ordinal = convert(Cint, i)
        handle_ref = Ref{Cint}()
        @cucall(:cuDeviceGet, (Ptr{Cint}, Cint), handle_ref, ordinal)
        new(ordinal, handle_ref[])
    end
end

function name(dev::CuDevice)
    const buflen = 256
    buf = Array(Cchar, buflen)
    @cucall(:cuDeviceGetName, (Ptr{Cchar}, Cint, Cint),
                              buf, buflen, dev.handle)
    return bytestring(pointer(buf))
end

function totalmem(dev::CuDevice)
    mem_ref = Ref{Csize_t}()
    @cucall(:cuDeviceTotalMem, (Ptr{Csize_t}, Cint), mem_ref, dev.handle)
    return mem_ref[]
end

function attribute(dev::CuDevice, attrcode::Integer)
    value_ref = Ref{Csize_t}()
    @cucall(:cuDeviceGetAttribute, (Ptr{Cint}, Cint, Cint),
                                   value_ref, attrcode, dev.handle)
    return value_ref[]
end

function capability(dev::CuDevice)
    major_ref = Ref{Cint}()
    minor_ref = Ref{Cint}()
    @cucall(:cuDeviceComputeCapability, (Ptr{Cint}, Ptr{Cint}, Cint),
                                        major_ref, minor_ref, dev.handle)
    return VersionNumber(major_ref[], minor_ref[])
end

# NOTE: keep this in sync with the architectures supported by NVPTX
#       (see lib/Target/NVPTX/NVPTXGenSubtargetInfo.inc)
const architectures = [
    (v"2.0", "sm_20"),
    (v"2.1", "sm_21"),
    (v"3.0", "sm_30"),
    (v"3.5", "sm_35"),
    (v"5.0", "sm_50") ]

function architecture(dev::CuDevice)
    # TODO: this is used both when compiling natively, and when addressing NVCC
    #       check both their restrictions individually
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
end

function list_devices()
    cnt = devcount()
    if cnt == 0
        println("No CUDA-capable device found.")
        return
    end

    for i = 0:cnt-1
        dev = CuDevice(i)
        nam = name(dev)
        tmem = iround(totalmem(dev) / (1024^2))
        cap = capability(dev)

        println("device[$i]: $(nam), capability $(cap.major).$(cap.minor), total mem = $tmem MB")
    end
end

