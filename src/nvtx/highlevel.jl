#
# domains
#

export Domain, domain

struct Domain
    handle::nvtxDomainHandle_t

    function Domain(name::String)
        info("Creating domain $name")
        handle = nvtxDomainCreateA(name)
        new(handle)
    end
end

Base.unsafe_convert(::Type{nvtxDomainHandle_t}, dom::Domain) = dom.handle

unsafe_destroy!(dom::Domain) = nvtxDomainDestroy(dom)

function Domain(f::Function, name::String)
    dom = Domain(name)
    f(dom)
    unsafe_destroy!(dom)
end


#
# markers
#

export mark

mark(msg::String) = nvtxMarkA(msg)


#
# ranges
#

export Range, start_range, stop_range, @range

struct Range
    id::nvtxRangeId_t
end

Base.convert(::Type{nvtxRangeId_t}, range::Range) = range.id

"""
    start_range(msg)

Create and start a new range. The range is not automatically stopped, use
[`end_range(::Range)`](@ref) for that.

Use this API if you need overlapping ranges, for scope-based use [`@range`](@ref) instead.
"""
start_range(msg::String) = nvtxRangeStartA(msg)
end_range(r::Range) = nvtxRangeEnd(r)

push_range(msg::String) = nvtxRangePushA(msg)
pop_range() = nvtxRangePop()

"""
    @range "msg" ex

Create a new range and execute `ex`. The range is popped automatically afterwards.

See also: [`range`](@ref)
"""
macro range(msg, ex)
    quote
        push_range($(esc(msg)))
        local ret = $(esc(ex))
        pop_range()
        ret
    end
end
