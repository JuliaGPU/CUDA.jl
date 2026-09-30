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

# same, but callable from any thread
function worker_synchronize(obj::SyncObject)
    context!(obj isa CuContext ? obj : obj.ctx) do
        unchecked_synchronize(obj)
    end
end

function synchronize_object(obj::SyncObject; blocking::Bool, spin::Bool)
    # there is no way to poll an entire context (querying the legacy stream does not cover
    # non-blocking streams)
    isdone = obj isa CuContext ? nothing : CUDACore.isdone

    # if we're about to wait, now may be a good time for a GC pause
    if isdone === nothing || !isdone(obj)
        maybe_collect(true)
    end

    res = if !blocking && use_nonblocking_synchronization
        # if the object was found to be done, synchronize again to report errors
        @something(cooperative_wait(worker_synchronize, obj; isdone, spin),
                   unchecked_synchronize(obj))::CUresult
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
    synchronize_object(stream; blocking, spin)
    check_exceptions()
end

synchronize(event::CuEvent; blocking::Bool=false, spin::Bool=true) =
    synchronize_object(event; blocking, spin)
