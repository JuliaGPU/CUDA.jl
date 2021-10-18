# Events for timing

export CuEvent, record, synchronize, elapsed


@enum_without_prefix CUevent_flags_enum CU_

"""
    CuEvent()

Create a new CUDA event.
"""
mutable struct CuEvent
    handle::CUevent
    ctx::CuContext

    function CuEvent(flags=EVENT_DEFAULT)
        handle_ref = Ref{CUevent}()
        cuEventCreate(handle_ref, flags)

        ctx = current_context()
        obj = new(handle_ref[], ctx)
        finalizer(unsafe_destroy!, obj)
        return obj
    end
end

function unsafe_destroy!(e::CuEvent)
    @finalize_in_ctx e.ctx cuEventDestroy_v2(e)
end

Base.unsafe_convert(::Type{CUevent}, e::CuEvent) = e.handle

Base.:(==)(a::CuEvent, b::CuEvent) = a.handle == b.handle
Base.hash(e::CuEvent, h::UInt) = hash(e.handle, h)

"""
    record(e::CuEvent, [stream::CuStream])

Record an event on a stream.
"""
record(e::CuEvent, stream::CuStream=stream()) =
    cuEventRecord(e, stream)

"""
    synchronize(e::CuEvent)

Waits for an event to complete.
"""
synchronize(e::CuEvent) = cuEventSynchronize(e)

"""
    isdone(e::CuEvent)

Return `false` if there is outstanding work preceding the most recent
call to `record(e)` and `true` if all captured work has been completed.
"""
function isdone(e::CuEvent)
    res = unsafe_cuEventQuery(e)
    if res == ERROR_NOT_READY
        return false
    elseif res == SUCCESS
        return true
    else
        throw_api_error(res)
    end
end

"""
    wait(e::CuEvent, [stream::CuStream])

Make a stream wait on a event. This only makes the stream wait, and not the host; use
[`synchronize(::CuEvent)`](@ref) for that.
"""
wait(e::CuEvent, stream::CuStream=stream()) =
    cuStreamWaitEvent(stream, e, 0)

"""
    elapsed(start::CuEvent, stop::CuEvent)

Computes the elapsed time between two events (in seconds).
"""
function elapsed(start::CuEvent, stop::CuEvent)
    time_ref = Ref{Cfloat}()
    cuEventElapsedTime(time_ref, start, stop)
    return time_ref[]/1000
end

"""
    @elapsed ex

A macro to evaluate an expression, discarding the resulting value, instead returning the
number of seconds it took to execute on the GPU, as a floating-point number.
"""
macro elapsed(ex)
    quote
        t0, t1 = CuEvent(), CuEvent()
        record(t0)
        $(esc(ex))
        record(t1)
        synchronize(t1)
        elapsed(t0, t1)
    end
end
