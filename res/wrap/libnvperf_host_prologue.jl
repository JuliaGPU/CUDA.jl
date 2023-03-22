# outlined functionality to avoid GC frame allocation
@noinline function throw_api_error(res)
    throw(NVPAError(res))
end

macro check(ex, errs...)
    quote
        res = $(esc(ex))
        if res != NVPA_STATUS_SUCCESS
            throw_api_error(res)
        end

        nothing
    end
end

macro NVPA_STRUCT_SIZE(type, lastfield)
    type = esc(type)
    lastfield = QuoteNode(lastfield)
    quote
        $struct_size($type, $lastfield)
    end
end