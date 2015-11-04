# Basic library loading and API calling

import Base: get

export
    CUDA_VENDOR,
    @cucall


# Early library loading
# TODO: allow overriding the loaded library using ENV
function load_library()
    try
        # TODO: thorough check of vendor?
        return (Libdl.dlopen("libcuda"), "NVIDIA")
    end

    try
        return (Libdl.dlopen("libocelot"), "Ocelot")
    end

    error("Could not load CUDA (or any compatible) library")
end
const (libcuda, CUDA_VENDOR) = load_library()

# API call wrapper
macro cucall(f, argtypes, args...)
    # Escape the tuple of arguments, making sure it is evaluated in caller scope
    # (there doesn't seem to be inline syntax like `$(esc(argtypes))` for this)
    esc_args = [esc(arg) for arg in args]

    quote
        api_function = resolve($f)
        status = ccall(Libdl.dlsym(libcuda, api_function), Cint, $(esc(argtypes)), $(esc_args...))
        if status != 0
            err = CuError(Int(status))
            throw(err)
        end
    end
end

api_mapping = Dict{Symbol,Symbol}()
resolve(f::Symbol) = get(api_mapping, f, f)

function initialize_api()
    # Create mapping for versioned API calls
    if haskey(ENV, "CUDA_FORCE_API_VERSION")
        api_version = ENV["CUDA_FORCE_API_VERSION"]
    else
        api_version = driver_version()
    end
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
end

function driver_version()
    version_ref = Ref{Cint}()
    @cucall(:cuDriverGetVersion, (Ptr{Cint},), version_ref)
    return version_ref[]
end
