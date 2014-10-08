# Load & initialize CUDA driver

const libcuda = dlopen("libcuda")

macro cucall(f, argtypes, args...)
	quote
		g = haskey(api_dict, $f) ? api_dict[$f] : $f
		_curet = ccall(dlsym(libcuda, g), Cint, $argtypes, $(args...))
		if _curet != 0
			err = CuDriverError(int(_curet))
			message = description(err)
			throw(err)
		end
	end
end

function initialize(api_version::Int)
	populate_api_dict(api_version)
	@cucall(:cuInit, (Cint,), 0)
end

# Emulate device synchronization by synchronizing stream 0
synchronize() = @cucall("cuStreamSynchronize", (Ptr{Void},), 0)


# Get driver version

function driver_version()
	a = Cint[0]
	@cucall(:cuDriverGetVersion, (Ptr{Cint},), a)
	return int(a[1])
end


# box a variable into array

cubox{T}(x::T) = T[x]


# create dict for ambiguous api calls

global api_dict = Dict{Symbol,Symbol}()
function populate_api_dict(api_version::Int)
	if api_version >= 3020
		api_dict[:cuDeviceTotalMem]   = :cuDeviceTotalMem_v2
		api_dict[:cuCtxCreate]        = :cuCtxCreate_v2
		api_dict[:cuMemAlloc]         = :cuMemAlloc_v2
		api_dict[:cuMemcpyHtoD]       = :cuMemcpyHtoD_v2
		api_dict[:cuMemcpyDtoH]       = :cuMemcpyDtoH_v2
		api_dict[:cuMemFree]          = :cuMemFree_v2
		api_dict[:cuModuleGetGlobal]  = :cuModuleGetGlobal_v2
		api_dict[:cuMemsetD32]        = :cuMemsetD32_v2
	end
	if api_version >= 4000
		api_dict[:cuCtxDestroy]       = :cuCtxDestroy_v2
		api_dict[:cuCtxPushCurrent]   = :cuCtxPushCurrent_v2
		api_dict[:cuCtxPopCurrent]    = :cuCtxPopCurrent_v2
	end
end


initialize(5050)
