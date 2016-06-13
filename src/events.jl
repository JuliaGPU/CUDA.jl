# Events for timing

import Base: unsafe_convert

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

unsafe_convert(::Type{CuEvent_t}, e::CuEvent) = e.handle

record(e::CuEvent, stream=default_stream()) = 
    @apicall(:cuEventRecord, (CuEvent_t, CuStream_t), e.handle, stream.handle)

synchronize(e::CuEvent) = @apicall(:cuEventSynchronize, (CuEvent_t,), e.handle)

function elapsed(start::CuEvent, stop::CuEvent)
    time_ref = Ref{Cfloat}()
    @apicall(:cuEventElapsedTime, 
        (Ptr{Cfloat}, CuEvent_t, CuEvent_t), 
        time_ref, start.handle, stop.handle)
    return time_ref[]
end

destroy(e::CuEvent) = @apicall(:cuEventDestroy, (CuEvent_t,), e.handle)
