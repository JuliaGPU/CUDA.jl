# Events for timing

export CuEvent, record, synchronize, elapsed, @elapsed


const CuEvent_t = Ptr{Void}

"""
    CuEvent()

Create a new CUDA event.
"""
type CuEvent
    handle::CuEvent_t
    ctx::CuContext

    function CuEvent()
        handle_ref = Ref{CuEvent_t}()
        @apicall(:cuEventCreate, (Ptr{CuEvent_t}, Cuint), handle_ref, 0)

        ctx = CuCurrentContext()
        obj = new(handle_ref[], ctx)
        finalizer(obj, unsafe_destroy!)
        return obj
    end 
end

function unsafe_destroy!(e::CuEvent)
    if isvalid(e.ctx)
        @trace("Finalizing CuEvent object at $(Base.pointer_from_objref(e))")
        @apicall(:cuEventDestroy, (CuEvent_t,), e)
    else
        @trace("Skipping finalizer for CuEvent object at $(Base.pointer_from_objref(e))) because context is no longer valid.")
    end
end

Base.unsafe_convert(::Type{CuEvent_t}, e::CuEvent) = e.handle

Base.:(==)(a::CuEvent, b::CuEvent) = a.handle == b.handle
Base.hash(e::CuEvent, h::UInt) = hash(e.handle, h)

"""
    record(e::CuEvent, stream=CuDefaultStream())

Record an event on a stream.
"""
record(e::CuEvent, stream::CuStream=CuDefaultStream()) =
    @apicall(:cuEventRecord, (CuEvent_t, CuStream_t), e, stream)

"""
    synchronize(e::CuEvent)

Waits for an event to complete.
"""
synchronize(e::CuEvent) = @apicall(:cuEventSynchronize, (CuEvent_t,), e)

"""
    elapsed(start::CuEvent, stop::CuEvent)

Computes the elapsed time between two events (in seconds).
"""
function elapsed(start::CuEvent, stop::CuEvent)
    time_ref = Ref{Cfloat}()
    @apicall(:cuEventElapsedTime, (Ptr{Cfloat}, CuEvent_t, CuEvent_t),
                                  time_ref, start, stop)
    return time_ref[]/1000
end

"""
    @elapsed stream ex
    @elapsed ex

A macro to evaluate an expression, discarding the resulting value, instead returning the
number of seconds it took to execute on the GPU, as a floating-point number.
"""
macro elapsed(stream, ex)
    quote
        t0, t1 = CuEvent(), CuEvent()
        record(t0, $stream)
        $(esc(ex))
        record(t1, $stream)
        synchronize(t1)
        elapsed(t0, t1)
    end
end
macro elapsed(ex)
    quote
        @elapsed(CuDefaultStream(), $(esc(ex)))
    end
end
