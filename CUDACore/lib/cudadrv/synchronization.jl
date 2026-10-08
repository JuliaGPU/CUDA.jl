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
worker_isdone((obj, ctx)) = poll(obj)

# whether the object is done, without taking locks, so that this can be called from any
# thread. (`isdone` on an event also learns what the work it covers is.)
poll(obj::CuStream) = isdone(obj)
poll(obj::CuEvent) = unsafe_isdone(obj)

function synchronize_object(obj::SyncObject; blocking::Bool, spin::Bool)
    obj isa CuEvent || check_capture(obj)

    # there is no way to poll an entire context (querying the legacy stream does not cover
    # non-blocking streams)
    pollable = !(obj isa CuContext)
    nonblocking = !blocking && use_nonblocking_synchronization

    # if polling finds the object to be done, there's no need to synchronize again: `isdone`
    # reports errors, and a successful query counts as synchronization (e.g., for accessing
    # unified memory). doing so anyway would risk blocking the thread, if another task
    # submitted work in the meantime. polling here, instead of leaving that to
    # `cooperative_wait`, lets short operations skip the GC pause and draining. blocking
    # synchronization doesn't poll, as a query costs about as much as synchronizing an idle
    # stream.
    res = if nonblocking && pollable && spin && poll(obj)
        synchronize_completed(obj)
    else
        # if we're about to wait, now may be a good time for a GC pause
        maybe_collect(true)

        # release resources that finalizers retired, before waiting (so that the GPU can
        # process the frees in the meantime) and after (as finalizers may have run meanwhile)
        drain_retired(ALLOC_DRAIN_LIMIT)

        if nonblocking
            ctx = obj isa CuContext ? obj : obj.ctx === nothing ? context() : obj.ctx
            res = cooperative_wait(worker_synchronize, (obj, ctx);
                                   isdone = pollable ? worker_isdone : nothing, spin)
            if res === nothing && pollable
                synchronize_completed(obj)
            else
                something(res, SUCCESS)::CUresult
            end
        else
            unchecked_synchronize(obj)
        end
    end

    if res != SUCCESS
        throw_api_error(res)
    end
    drain_retired(ALLOC_DRAIN_LIMIT)
    return
end

# XXX: compute-sanitizer doesn't treat a successful query of an event as synchronization, and
#      reports races with the work that the event ordered (#3346). so when synchronizing an
#      event finds it done by polling, synchronize it too. that only blocks if the event was
#      recorded again since. `isdone` also does so when it learns that tracked work has
#      completed. other queries of events, e.g. the ones that gate releasing held resources
#      and reusing cached memory, aren't covered.
#      synchronizing is prohibited while another thread captures in global mode, even though
#      it doesn't affect that capture, so relax the capture mode.
synchronize_completed(event::CuEvent) =
    relaxed_capture_mode(() -> unchecked_synchronize(event))::CUresult

# polling a stream doesn't flush the output of its kernels (from `printf`), as synchronizing
# does. synchronizing it anyway would wait for work submitted since, so flush otherwise.
synchronize_completed(stream::CuStream) =
    flush_output(stream.ctx === nothing ? context() : stream.ctx)

# synchronizing an event that was never recorded flushes kernel output without waiting for
# any work. that's also allowed while capturing, once relaxing the capture mode.
function flush_output(ctx::CuContext)
    context!(ctx) do
        handle = Ref{CUevent}()
        cuEventCreate(handle, EVENT_DISABLE_TIMING)
        try
            relaxed_capture_mode(() -> unchecked_cuEventSynchronize(handle[]))::CUresult
        finally
            cuEventDestroy_v2(handle[])
        end
    end
end

function device_synchronize(; blocking::Bool=false, spin::Bool=true)
    synchronize_object(context(); blocking, spin)
    check_exceptions()
end

function synchronize(stream::CuStream=stream(); blocking::Bool=false, spin::Bool=true)
    order = stream.order
    epoch = if order === nothing
        nothing
    else
        # (a stream that is being captured can't be synchronized, so don't close its epoch)
        check_capture(stream)
        close_epoch!(order)
    end
    if stream.handle == CU_STREAM_PER_THREAD && !blocking && use_nonblocking_synchronization
        # the per-thread stream is specific to the calling thread, so it can't be
        # synchronized from a worker thread. wait for an event recorded on it instead.
        event = CuEvent(EVENT_DISABLE_TIMING)
        try
            cuEventRecord(event, stream)
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

# the event's lock isn't held while waiting, so that other tasks can record or query it in
# the meantime. what the wait covers is only learned if the event hasn't been recorded again.
function synchronize(event::CuEvent; blocking::Bool=false, spin::Bool=true)
    source = Base.@lock event.lock event.source
    synchronize_object(event; blocking, spin)
    source === nothing || Base.@lock event.lock mark_completed!(event, source)
    return
end
