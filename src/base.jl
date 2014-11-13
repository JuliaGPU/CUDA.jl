# Basic library loading and API calling

import Base: get

export
	@cucall


const libcuda = dlopen("libocelot")

# API call wrapper
macro cucall(f, argtypes, args...)
	quote
		api_function = resolve($f)
		status = ccall(dlsym(libcuda, api_function), Cint, $argtypes, $(args...))
		if status != 0
			err = CuError(int(status))
			throw(err)
		end
	end
end

api_mapping = Dict{Symbol,Symbol}()
resolve(f::Symbol) = get(api_mapping, f, f)

function initialize()
	# Create mapping for versioned API calls
	api_version = driver_version()
	global api_mapping
	if api_version >= 3020
		api_mapping[:cuDeviceTotalMem]   = :cuDeviceTotalMem_v2
		api_mapping[:cuCtxCreate]        = :cuCtxCreate_v2
		api_mapping[:cuMemAlloc]         = :cuMemAlloc_v2
		api_mapping[:cuMemcpyHtoD]       = :cuMemcpyHtoD_v2
		api_mapping[:cuMemcpyDtoH]       = :cuMemcpyDtoH_v2
		api_mapping[:cuMemFree]          = :cuMemFree_v2
		api_mapping[:cuModuleGetGlobal]  = :cuModuleGetGlobal_v2
		api_mapping[:cuMemsetD32]        = :cuMemsetD32_v2
	end
	if api_version >= 4000
		api_mapping[:cuCtxDestroy]       = :cuCtxDestroy_v2
		api_mapping[:cuCtxPushCurrent]   = :cuCtxPushCurrent_v2
		api_mapping[:cuCtxPopCurrent]    = :cuCtxPopCurrent_v2
	end

	# Initialize the driver
	@cucall(:cuInit, (Cint,), 0)

	ccall(:jl_init_ptx_codegen, Void, (String, String),
		  "nvptx64-nvidia-cuda", "sm_20")
end

function driver_version()
	version_box = ptrbox(Cint)
	@cucall(:cuDriverGetVersion, (Ptr{Cint},), version_box)
	return ptrunbox(version_box)
end

# Box a variable into an array for ccall() passing
ptrbox(T::Type) = Array(T, 1)
ptrbox(T::Type, val) = T[val]
ptrunbox{T}(box::Array{T, 1}) = box[1]
ptrunbox{T}(box::Array{T, 1}, desttype::Type) = convert(desttype, ptrunbox(box))
