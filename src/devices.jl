# CUDA CuDevice management

immutable CuDevice
	ordinal::Cint
	handle::Cint

	function CuDevice(i::Integer)
		ordinal = convert(Cint, i)
		a = Cint[0]
		@cucall(:cuDeviceGet, (Ptr{Cint}, Cint), a, ordinal)
		handle = a[1]
		new(ordinal, handle)		
	end
end

function name(dev::CuDevice)
	const buflen = 256
	buf = Array(Cchar, buflen)
	@cucall(:cuDeviceGetName, (Ptr{Cchar}, Cint, Cint), buf, buflen, dev.handle)
	bytestring(pointer(buf))
end

function totalmem(dev::CuDevice)
	a = Csize_t[0]
	@cucall(:cuDeviceTotalMem, (Ptr{Csize_t}, Cint), a, dev.handle)
	return int(a[1])
end

function attribute(dev::CuDevice, attrcode::Integer)
	a = Cint[0]
	@cucall(:cuDeviceGetAttribute, (Ptr{Cint}, Cint, Cint), a, attrcode, dev.handle)
	return int(a[1])
end

function capability(dev::CuDevice)
	major = Cint[0]
	minor = Cint[0]
	@cucall(:cuDeviceComputeCapability, (Ptr{Cint}, Ptr{Cint}, Cint), major, minor, dev.handle)
	return VersionNumber(major[1], minor[1])
end

function devcount()
	# Get the number of CUDA-capable CuDevices
	a = Cint[0]
	@cucall(:cuDeviceGetCount, (Ptr{Cint},), a)
	return int(a[1])
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

