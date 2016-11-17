# Events for timing

export CuEvent, record, synchronize, elapsed


typealias CuEvent_t Ptr{Void}

type CuEvent
    handle::CuEvent_t
    ctx::CuContext

    function CuEvent()
        handle_ref = Ref{CuEvent_t}()
        @apicall(:cuEventCreate, (Ptr{CuEvent_t}, Cuint), handle_ref, 0)

        ctx = CuCurrentContext()
        obj = new(handle_ref[], ctx)
        block_finalizer(obj, ctx)
        finalizer(obj, finalize)
        return obj
    end 
end

function finalize(e::CuEvent)
    trace("Finalizing CuEvent at $(Base.pointer_from_objref(e))")
    @apicall(:cuEventDestroy, (CuEvent_t,), e)
    unblock_finalizer(e, e.ctx)
end

Base.unsafe_convert(::Type{CuEvent_t}, e::CuEvent) = e.handle

Base.:(==)(a::CuEvent, b::CuEvent) = a.handle == b.handle
Base.hash(e::CuEvent, h::UInt) = hash(e.handle, h)

record(e::CuEvent, stream::CuStream=CuDefaultStream()) =
    @apicall(:cuEventRecord, (CuEvent_t, CuStream_t), e, stream)

synchronize(e::CuEvent) = @apicall(:cuEventSynchronize, (CuEvent_t,), e)

function elapsed(start::CuEvent, stop::CuEvent)
    time_ref = Ref{Cfloat}()
    @apicall(:cuEventElapsedTime, (Ptr{Cfloat}, CuEvent_t, CuEvent_t),
                                  time_ref, start, stop)
    return time_ref[]
end
