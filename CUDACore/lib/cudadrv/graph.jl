## stream capture
#
# the graph objects, and the integration of stream capture with the memory allocator, are
# implemented in `src/graph.jl`. here, we only provide queries and helpers that other
# driver-level code needs.

export capture_status, is_capturing, CaptureError

@enum_without_prefix visibility=:public CUstreamCaptureMode CU_
@enum_without_prefix visibility=:public CUstreamCaptureStatus CU_

"""
    capture_status([stream::CuStream])

Return the capture status of a stream as a named tuple `(status, id)`, where `status` is a
`CUstreamCaptureStatus` and `id` is the unique identifier of the capture sequence the stream
is part of (or `nothing` when the stream is not being captured).
"""
function capture_status(stream::CuStream=stream())
    status_ref = Ref{CUstreamCaptureStatus}()
    id_ref = Ref{UInt64}()
    cuStreamGetCaptureInfo(stream, status_ref, id_ref)
    return (status=status_ref[],
            id=(status_ref[] == STREAM_CAPTURE_STATUS_ACTIVE ? id_ref[] : nothing))
end

"""
    is_capturing([stream::CuStream])

Return whether a stream is being captured into a graph.
"""
@inline function is_capturing(stream::CuStream=stream())
    status = Ref{CUstreamCaptureStatus}()
    cuStreamIsCapturing(stream, status)
    return status[] != STREAM_CAPTURE_STATUS_NONE
end

# the number of captures in progress using `capture`. many operations need to behave
# differently while capturing, e.g., by not releasing memory, but checking for that on
# every operation would be too expensive. captures started directly through the driver API
# are not counted, and are not supported by CUDA.jl's memory management.
const active_captures = Threads.Atomic{Int}(0)

# whether `stream` is being captured, cheaply checking whether any capture is in progress
@inline in_capture(stream::CuStream) = active_captures[] > 0 && is_capturing(stream)

# some API calls, like synchronizing an unrelated stream, are prohibited while a stream
# is being captured, even though they don't interfere with that capture. relaxing the
# capture mode of the thread makes it possible to call them. `f` must not yield, because
# the capture mode is a property of the thread.
function relaxed_capture_mode(f)
    mode = Ref(STREAM_CAPTURE_MODE_RELAXED)
    cuThreadExchangeStreamCaptureMode(mode)
    try
        f()
    finally
        cuThreadExchangeStreamCaptureMode(mode)
    end
end

"""
    CaptureError(msg)

An operation was attempted that isn't supported while capturing a graph, e.g., waiting for
the GPU, which is not possible as captured operations only execute when the graph is
launched.
"""
struct CaptureError <: Exception
    msg::String
end

Base.showerror(io::IO, err::CaptureError) = print(io, "CaptureError: ", err.msg)

# waiting for an object can't be done when it involves work that is being captured
function check_capture(obj::Union{CuStream,CuContext})
    active_captures[] == 0 && return
    if obj isa CuContext
        # synchronizing a context waits for every stream in it, including captured ones
        throw(CaptureError("""cannot synchronize the device while a graph is being captured.
                              That would wait for the captured operations, which only execute when the graph is launched,
                              and invalidate the capture. Synchronize a stream instead, or wait for the capture to finish."""))
    elseif is_capturing(obj)
        throw(CaptureError("""cannot wait for the GPU while capturing a graph.
                              Captured operations only execute when the graph is launched, so it is not possible to wait for their results,
                              or to access GPU memory from the CPU (e.g., by copying to an `Array`, or by indexing an array)."""))
    end
end


## capture scopes
#
# tasks that are spawned while capturing a graph, e.g., by a library that parallelizes its
# work, take part in the capture: their operations are captured on the same stream as the
# operations of the capturing task. that stream orders operations as they are submitted, so
# as long as tasks only depend on each other through the usual means (waiting for a task,
# or synchronizing through locks, channels, etc.), the captured graph orders operations at
# least as strictly as executing them would have. every submission is performed while
# holding a lock, so that operations consisting of multiple submissions (like library calls)
# aren't interleaved. the tasks need to have finished before the capture ends.

@enum CaptureScopeState::UInt8 SCOPE_OPEN SCOPE_CLOSING SCOPE_CLOSED

mutable struct CaptureScope
    Base.@atomic state::CaptureScopeState
    const stream::CuStream
    const context::CuContext
    const owner::Task
    # tasks other than the owner that submitted operations
    const participants::Base.IdSet{Task}
    const participants_lock::Threads.SpinLock
    # serializes submissions, see `capture_submission`
    const lock::ReentrantLock

    function CaptureScope(stream::CuStream, ctx::CuContext)
        scope = new(SCOPE_OPEN, stream, ctx, current_task(), Base.IdSet{Task}(),
                    Threads.SpinLock(), ReentrantLock())
        Threads.atomic_add!(capture_scopes, 1)
        finalizer(scope) do _
            Threads.atomic_sub!(capture_scopes, 1)
        end
    end
end

const capture_scope = ScopedValues.ScopedValue{Union{Nothing,CaptureScope}}(nothing)

# the number of capture scopes that are still reachable. tasks that were spawned during a
# capture keep its scope alive, also after the capture has ended, so that they can detect
# that they shouldn't perform GPU operations anymore. as long as there are none, looking up
# the scope of the current task can be skipped.
const capture_scopes = Threads.Atomic{Int}(0)

# the capture scope of the current task, if any
@inline function current_capture_scope()
    capture_scopes[] == 0 && return nothing
    return capture_scope[]
end

@noinline function scope_closed_error()
    throw(CaptureError("""this task was started while capturing a graph, which has ended.
                          Tasks that perform GPU operations while capturing a graph need to finish before the capture ends."""))
end

# the stream that tasks that are part of a capture use
@noinline function scoped_stream(scope::CaptureScope, ctx::CuContext)
    (Base.@atomic :acquire scope.state) == SCOPE_OPEN || scope_closed_error()
    scope.context == ctx ||
        throw(CaptureError("cannot switch to another device while capturing a graph"))
    return scope.stream
end

"""
    capture_submission(f, [stream::CuStream])

Call `f`, which submits operations to `stream`, as a single submission of the capture that
the current task is part of, if any. Tasks that are spawned while capturing a graph take
part in the capture, and need to perform their submissions this way so that they are
captured in a consistent order. Calls can be nested.
"""
@inline function capture_submission(f, stream::Union{Nothing,CuStream}=nothing)
    scope = current_capture_scope()
    scope === nothing && return f()
    enter_submission(scope, stream)
    try
        return f()
    finally
        unlock(scope.lock)
    end
end

@noinline function enter_submission(scope::CaptureScope, stream::Union{Nothing,CuStream})
    task = current_task()
    if task !== scope.owner
        # (registering before locking, so that tasks waiting to submit count as unfinished)
        @lock scope.participants_lock push!(scope.participants, task)
    end
    lock(scope.lock)
    try
        (Base.@atomic :acquire scope.state) == SCOPE_OPEN || scope_closed_error()
        if task !== scope.owner && stream !== nothing && stream != scope.stream
            throw(CaptureError("cannot submit operations to another stream from a task that is part of a graph capture"))
        end
    catch
        unlock(scope.lock)
        rethrow()
    end
    return
end

# stop admitting submissions to a capture, returning whether all tasks that submitted to it
# have finished.
function close_scope!(scope::CaptureScope)
    @lock scope.lock begin
        Base.@atomic :release scope.state = SCOPE_CLOSING
        @lock scope.participants_lock all(istaskdone, scope.participants)
    end
end
