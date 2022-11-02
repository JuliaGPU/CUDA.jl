# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(NVMLError(res))
end

const initialized = Ref(false)
function initialize_context()
    if !initialized[]
        res = unsafe_nvmlInitWithFlags(0)
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

macro check(ex)
    quote
        res = $(esc(ex))
        if res != NVML_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end
