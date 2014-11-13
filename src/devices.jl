# Device type and auxiliary functions

export
    devcount,
    CuDevice, name, totalmem, attribute, capability,
    list_devices


function devcount()
    count_box = ptrbox(Cint)
    @cucall(:cuDeviceGetCount, (Ptr{Cint},), count_box)
    ptrunbox(count_box)
end

immutable CuDevice
    ordinal::Cint
    handle::Cint

    function CuDevice(i::Integer)
        ordinal = convert(Cint, i)
        handle_box = ptrbox(Cint)
        @cucall(:cuDeviceGet, (Ptr{Cint}, Cint), handle_box, ordinal)
        new(ordinal, ptrunbox(handle_box))
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
    mem_box = ptrbox(Csize_t)
    @cucall(:cuDeviceTotalMem, (Ptr{Csize_t}, Cint), mem_box, dev.handle)
    return ptrunbox(mem_box)
end

function attribute(dev::CuDevice, attrcode::Integer)
    value_box = ptrbox(Csize_t)
    @cucall(:cuDeviceGetAttribute, (Ptr{Cint}, Cint, Cint),
                                   value_box, attrcode, dev.handle)
    return ptrunbox(value_box)
end

function capability(dev::CuDevice)
    major_box = ptrbox(Cint)
    minor_box = ptrbox(Cint)
    @cucall(:cuDeviceComputeCapability, (Ptr{Cint}, Ptr{Cint}, Cint),
                                        major_box, minor_box, dev.handle)
    return VersionNumber(ptrunbox(major_box), ptrunbox(minor_box))
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

