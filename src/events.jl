# Events for timing

export CuEvent, record, synchronize, elapsed


typealias CuEvent_t Ptr{Void}

immutable CuEvent
    handle::CuEvent_t

    function CuEvent()
        handle_ref = Ref{CuEvent_t}()
        @apicall(:cuEventCreate, (Ptr{CuEvent_t}, Cuint), handle_ref, 0)
        return new(handle_ref[])
    end 
end

Base.unsafe_convert(::Type{CuEvent_t}, e::CuEvent) = e.handle
Base.:(==)(a::CuEvent, b::CuEvent) = a.handle == b.handle

record(e::CuEvent, stream::CuStream=CuDefaultStream()) =
    @apicall(:cuEventRecord, (CuEvent_t, CuStream_t), e, stream)

synchronize(e::CuEvent) = @apicall(:cuEventSynchronize, (CuEvent_t,), e)

function elapsed(start::CuEvent, stop::CuEvent)
    time_ref = Ref{Cfloat}()
    @apicall(:cuEventElapsedTime, (Ptr{Cfloat}, CuEvent_t, CuEvent_t),
                                  time_ref, start, stop)
    return time_ref[]
end

destroy(e::CuEvent) = @apicall(:cuEventDestroy, (CuEvent_t,), e)
