release_now(event::CuEvent) = unsafe_destroy!(event)

# memory that was last used on a stream may be retired after that stream, and released
# after the stream has been destroyed. it then needs to wait for the work that was submitted
# to the stream, which is captured by an event recorded before destroying it.
const stream_disposal_lock = ReentrantLock()
function release_now(stream::CuStream)
  @lock stream_disposal_lock begin
    isvalid(stream) || return
    context!(stream.ctx) do
      event = CuEvent(EVENT_DISABLE_TIMING)
      cuEventRecord(event, stream)
      stream.final_event = event
      cuStreamDestroy_v2(stream)
    end
    Base.@atomic stream.valid = false
    # keep the final event alive until the work has finished, as memory that was last used
    # on the stream may be retired later on
    hold!(ReleaseAction(s -> (s.final_event = nothing), stream, nothing, false),
          stream.final_event)
  end
  return
end

# memory released after its stream was destroyed is released on a non-blocking stream that
# waits for the final event of that stream. (the legacy default stream would make other
# streams wait as well, which could deadlock.) it is created when a task starts using a
# context, so that releasing memory doesn't create streams (which can block).
const disposal_streams = Dict{CuContext,CUstream}()
const disposal_streams_lock = ReentrantLock()
function disposal_stream(ctx::CuContext)
  @lock disposal_streams_lock get!(disposal_streams, ctx) do
    # this is also when CUDA is first used, so start releasing retired resources periodically
    start_retired_drainer()
    context!(ctx) do
      handle = Ref{CUstream}()
      cuStreamCreate(handle, STREAM_NON_BLOCKING)
      handle[]
    end
  end
end

# a stream to release memory on that was last used on `stream` (during `generation`), along
# with its context. needs to be called with `stream_disposal_lock` held, which also keeps
# the stream from being handed to another task (see `claim_stream!`).
function release_stream(stream::CuStream, ctx::CuContext, generation::Int)
  if generation != CUDACore.generation(stream)
    # the stream has been handed to another task, so the work has finished. don't use the
    # stream, which would wait for the new owner's work, or end up in its capture.
    return disposal_stream(something(stream.ctx, ctx)), something(stream.ctx, ctx)
  end
  if isvalid(stream)
    return stream.handle, something(stream.ctx, ctx)
  end
  event = stream.final_event::Union{Nothing,CuEvent}
  if event === nothing
    # either the work on the stream has finished, or the stream was explicitly destroyed,
    # in which case the user should have made sure of that
    return disposal_stream(ctx), ctx
  end
  handle = disposal_stream(event.ctx)
  context!(event.ctx) do
    cuStreamWaitEvent(handle, event, 0)
  end
  return handle, event.ctx
end

# make the disposal stream wait for the work on `stream`
function after_on_disposal_stream(stream::CUstream, ctx::CuContext)
  disposal = disposal_stream(ctx)
  stream == disposal && return disposal
  event = CuEvent(EVENT_DISABLE_TIMING)
  cuEventRecord(event, stream)
  cuStreamWaitEvent(disposal, event, 0)
  return disposal
end

# the per-thread default stream is specific to the thread that used it, which isn't known
# when releasing memory, so memory last used on it is only released when memory is
# reclaimed, after synchronizing the context it was used in.
on_per_thread_stream(stream::CuStream) = stream.handle == CU_STREAM_PER_THREAD
function synchronize_and(f, ctx::CuContext)
  return x -> begin
    context!(device_synchronize, ctx)
    f(x)
  end
end


## objects that need to be kept alive while the GPU uses them

struct RetiredOwner
  owner::Any
  managed::Managed
end

release_owner(owner, managed::Managed) =
  release(m -> discard(RetiredOwner(owner, m)), managed)

function release_now(retired::RetiredOwner)
  stream = retired.managed.stream
  ctx = retired.managed.stream_ctx
  if on_per_thread_stream(stream)
    destroy_later(synchronize_and(identity, retired.managed.stream_ctx), retired.owner)
    return
  end
  event = try
    @lock stream_disposal_lock begin
      handle, ctx = release_stream(stream, ctx, retired.managed.generation)
      context!(ctx) do
        event = CuEvent(EVENT_DISABLE_TIMING)
        cuEventRecord(event, handle)
        event
      end
    end
  catch
    # the memory may still be in use, so keep the owner alive until reclaiming memory
    destroy_later(synchronize_and(identity, retired.managed.stream_ctx), retired.owner)
    rethrow()
  end
  hold!(ReleaseAction(identity, retired.owner, nothing, false), event)
  return
end
