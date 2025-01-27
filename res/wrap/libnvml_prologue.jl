# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(NVMLError(res))
end

const initialized = Ref(false)
function initialize_context()
    if !initialized[]
        res = unchecked_nvmlInitWithFlags(0)
        if res !== NVML_SUCCESS
            # NOTE: we can't call nvmlErrorString during initialization
            error("NVML could not be initialized ($res)")
        end
        atexit() do
            nvmlShutdown()
        end
        initialized[] = true
    end
end

@inline function check(f)
    res = f()
    if res != NVML_SUCCESS
        throw_api_error(res)
    end

    return
end

macro NVML_STRUCT_VERSION(typename, version)
    struct_typename = Symbol("nvml$(String(typename))_v$(version)_t")
    struct_type = getfield(__module__, struct_typename)
    struct_version = UInt32(sizeof(struct_type)) | (UInt32(version) << 24)
    return :($struct_version)
end
