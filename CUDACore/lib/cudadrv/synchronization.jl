# cooperative synchronization
#
# Waiting for the GPU should not block the calling thread, so that other Julia tasks can run
# in the meantime. See GPUToolbox's `cooperative_wait` for how this is implemented.

const use_nonblocking_synchronization =
    Preferences.@load_preference("nonblocking_synchronization", true)

const SyncObject = Union{CuContext, CuStream, CuEvent}

# perform a blocking synchronization, returning the result code
unchecked_synchronize(ctx::CuContext) = unchecked_cuCtxSynchronize()
unchecked_synchronize(stream::CuStream) = unchecked_cuStreamSynchronize(stream)
unchecked_synchronize(event::CuEvent) = unchecked_cuEventSynchronize(event)

# same, but callable from any thread. the context is passed explicitly, as the default
# streams do not have one.
function worker_synchronize((obj, ctx))
    context!(ctx) do
        unchecked_synchronize(obj)
    end
end
worker_isdone((obj, ctx)) = obj isa CuEvent ? unsafe_isdone(obj) : isdone(obj)

function synchronize_object(obj::SyncObject; blocking::Bool, spin::Bool)
    # there is no way to poll an entire context (querying the legacy stream does not cover
    # non-blocking streams)
    pollable = !(obj isa CuContext)

    # if we're about to wait, now may be a good time for a GC pause
    if !pollable || !(obj isa CuEvent ? unsafe_isdone(obj) : isdone(obj))
        maybe_collect(true)
    end

    res = if !blocking && use_nonblocking_synchronization
        ctx = obj isa CuContext ? obj : obj.ctx === nothing ? context() : obj.ctx
        # if polling found the object to be done, there's no need to synchronize again:
        # `isdone` reports errors, and a successful query counts as synchronization (e.g.,
        # for accessing unified memory). doing so anyway would risk blocking the thread,
        # if another task submitted work in the meantime.
        something(cooperative_wait(worker_synchronize, (obj, ctx);
                                   isdone = pollable ? worker_isdone : nothing, spin),
                  SUCCESS)::CUresult
    else
        unchecked_synchronize(obj)
    end

    if res != SUCCESS
        throw_api_error(res)
    end
    return
end

function device_synchronize(; blocking::Bool=false, spin::Bool=true)
    synchronize_object(context(); blocking, spin)
    check_exceptions()
end

function synchronize(stream::CuStream=stream(); blocking::Bool=false, spin::Bool=true)
    order = stream.order
    epoch = order === nothing ? nothing : close_epoch!(order)
    if stream.handle == CU_STREAM_PER_THREAD && !blocking && use_nonblocking_synchronization
        # the per-thread stream is specific to the calling thread, so it can't be
        # synchronized from a worker thread. wait for an event recorded on it instead.
        event = CuEvent(EVENT_DISABLE_TIMING)
        try
            record(event, stream)
            synchronize_object(event; blocking, spin)
        finally
            finalize(event)
        end
    else
        synchronize_object(stream; blocking, spin)
    end
    epoch === nothing || mark_completed!(order, epoch)
    check_exceptions()
end

function synchronize(event::CuEvent; blocking::Bool=false, spin::Bool=true)
    Base.@lock event.lock begin
        synchronize_object(event; blocking, spin)
        mark_completed!(event)
    end
    return
end
