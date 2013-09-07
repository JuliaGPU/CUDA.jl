
# Load & initialize CUDA driver

const libcuda = dlopen("libcuda")


macro cucall(f, argtypes, args...)
	quote
		_curet = ccall(dlsym(libcuda, $f), Cint, $argtypes, $(args...))
		if _curet != 0
			throw(DriverError(int(_curet)))
		end
	end
end

function initialize()
	@cucall(:cuInit, (Cint,), 0)
	println("CUDA Driver Initialized")
end

initialize()

