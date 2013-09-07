# CUDA Device management

function devcount()
	# Get the number of CUDA-capable devices
	a = Cint[0]
	@cucall(:cuDeviceGetCount, (Ptr{Cint},), a)
	return int(a[1])
end


immutable Device
	ordinal::Cint
	handle::Cint

	function Device(i::Int)
		ordinal = convert(Cint, i)
		a = Cint[0]
		@cucall(:cuDeviceGet, (Ptr{Cint}, Cint), a, ordinal)
		handle = a[1]
		new(ordinal, handle)		
	end
end

immutable Capability
	major::Int
	minor::Int
end

function name(dev::Device)
	const buflen = 256
	buf = Array(Cchar, buflen)
	@cucall(:cuDeviceGetName, (Ptr{Cchar}, Cint, Cint), buf, buflen, dev.handle)
	bytestring(pointer(buf))
end

function totalmem(dev::Device)
	a = Csize_t[0]
	@cucall(:cuDeviceTotalMem, (Ptr{Csize_t}, Cint), a, dev.handle)
	return int(a[1])
end

function attribute(dev::Device, attrcode::Integer)
	a = Cint[0]
	@cucall(:cuDeviceGetAttribute, (Ptr{Cint}, Cint, Cint), a, attrcode, dev.handle)
	return int(a[1])
end

capability(dev::Device) = Capability(attribute(dev, 75), attribute(dev, 76))

function list_devices()
	cnt = devcount()
	if cnt == 0
		println("No CUDA-capable device found.")
		return
	end

	for i = 0:cnt-1
		dev = Device(i)
		nam = name(dev)
		tmem = iround(totalmem(dev) / (1024^2))
		cap = capability(dev)

		println("Device[$i]: $(nam), capability $(cap.major).$(cap.minor), total mem = $tmem MB")
	end
end

