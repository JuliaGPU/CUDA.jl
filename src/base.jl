
# Load & initialize CUDA driver

const libcuda = dlopen("libcuda")


macro cucall(f, argtypes, args...)
	quote
		g = haskey(api_dict, $f) ? api_dict[$f] : $f
		_curet = ccall(dlsym(libcuda, g), Cint, $argtypes, $(args...))
		if _curet != 0
			err = CuDriverError(int(_curet))
			message = description(err)
			throw("CUDA driver error ($(err.code)) : $message")
		end
	end
end

function initialize(api_version::Int)
	populate_api_dict(api_version)
	@cucall(:cuInit, (Cint,), 0)
	println("CUDA Driver Initialized")
end


# Get driver version

function driver_version()
	a = Cint[0]
	@cucall(:cuDriverGetVersion, (Ptr{Cint},), a)
	return int(a[1])
end


# box a variable into array

cubox{T}(x::T) = T[x]


# create dict for ambiguous api calls

global api_dict = (Symbol => Symbol)[]
function populate_api_dict(api_version::Int)
	if api_version >= 3020
		api_dict[:cuDeviceTotalMem] = :cuDeviceTotalMem_v2
		api_dict[:cuCtxCreate] 		= :cuCtxCreate_v2
		api_dict[:cuMemAlloc] 		= :cuMemAlloc_v2
		api_dict[:cuMemcpyHtoD] 	= :cuMemcpyHtoD_v2
		api_dict[:cuMemcpyDtoH] 	= :cuMemcpyDtoH_v2
		api_dict[:cuMemFree]		= :cuMemFree_v2
	end
	if api_version >= 4000
		api_dict[:cuCtxDestroy]		= :cuCtxDestroy_v2
		api_dict[:cuCtxPushCurrent]	= :cuCtxPushCurrent_v2
		api_dict[:cuCtxPopCurrent]	= :cuCtxPopCurrent_v2
	end
end


initialize(5050)