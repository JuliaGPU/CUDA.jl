# Events for timing

export CuEvent, record, synchronize, elapsed

typealias CuEvent_t Ptr{Void}

immutable CuEvent
    handle::CuEvent_t

    function CuEvent()
        handle_ref = Ref{CuEvent_t}()
        @cucall(:cuEventCreate, (Ptr{CuEvent_t}, Cuint), handle_ref, 0)
        return new(handle_ref[])
    end 
end

record(ce::CuEvent, stream=default_stream()) = 
    @cucall(:cuEventRecord, (CuEvent_t, Ptr{Void}), ce.handle, stream.handle)

synchronize(ce::CuEvent) = @cucall(:cuEventSynchronize, (Ptr{Void},), ce.handle)

function elapsed(start::CuEvent, stop::CuEvent)
    ms_ref = Ref{Cfloat}()
    @cucall(:cuEventElapsedTime, 
        (Ptr{Cfloat}, CuEvent_t, CuEvent_t), 
        ms_ref, start.handle, stop.handle)
    return ms_ref[]
end

destroy(ce::CuEvent) = @cucall(:cuEventDestroy, (CuEvent_t,), ce.handle)
