# Stream management

export CuStream, default_stream, legacy_stream, per_thread_stream,
       unique_id, priority, priority_range, synchronize, device_synchronize

# What the host knows about the work submitted to a stream, so that memory moving between
# streams only needs to be ordered when that's actually necessary (see `take_ownership!`).
# The stream's work is divided in epochs: memory accesses are stamped with the current
# epoch, and recording an event or synchronizing the stream closes it. An event or
# synchronization thus covers all accesses stamped with an epoch up to and including the
# one it closed. Recording an event on a stream that is being captured only adds a node to
# the graph, so it doesn't close anything.
mutable struct StreamOrder
    # the epoch accesses are currently stamped with
    Base.@atomic epoch::UInt64

    # all accesses stamped with this epoch, or an earlier one, have completed
    Base.@atomic completed::UInt64

    # the task submitting work to this stream, and whether other tasks do so too. a stream
    # stays shared: tasks that activated it with `stream!` may still use it after the task
    # it was created for has finished and it has been recycled.
    Base.@atomic owner::Union{Nothing,WeakRef}
    Base.@atomic shared::Bool

    # work submitted to this stream from now on is ordered after the accesses to other
    # streams up to these epochs, because the stream waited for an event covering them.
    # weakly keyed, as there's nothing to learn about streams that are gone.
    const waited::WeakKeyDict{StreamOrder,UInt64}

    # an event to make other streams wait for this one, created when first needed. the lock
    # is held from recording the event until the other stream waited for it. (a `CuEvent`,
    # which is defined after streams.)
    event::Any

    # while `capture` is capturing on the stream, recording an event on it would only add a
    # node to the graph. instead, other streams wait for an event that was recorded right
    # before the capture began, which covers the work that was submitted to the stream
    # before. it's set while holding the lock, so that no event is recorded on the stream
    # after the capture began (see `capture(; stream)`). also a `CuEvent`, or `nothing`.
    capture_event::Any

    const lock::ReentrantLock

    StreamOrder() = new(1, 0, nothing, false, WeakKeyDict{StreamOrder,UInt64}(), nothing,
                        nothing, ReentrantLock())
end

current_epoch(order::StreamOrder) = Base.@atomic order.epoch

# close the current epoch, returning it when the operation about to be submitted (an event
# or synchronization) will cover all accesses stamped with it. the epoch needs to be closed
# *before* submitting that operation, so that accesses stamped after submitting their own
# work aren't covered. however, an access may also be stamped when taking a pointer, before
# submitting the work that uses it. only the task submitting work to the stream knows that
# has happened, so an epoch closed by another task, or when several tasks submit work to
# the stream, doesn't cover anything.
function close_epoch!(order::StreamOrder)
    epoch = (Base.@atomic order.epoch += 1) - 1
    owner = Base.@atomic order.owner
    if Base.@atomic(order.shared) || (owner !== nothing && owner.value !== current_task())
        return nothing
    end
    return epoch
end

# note that the current task submits work to the stream. a task that has finished can't be
# about to submit work anymore, so its stream can be handed to another task, e.g., when
# streams are recycled, without considering it shared.
function bind!(order::StreamOrder)
    task = current_task()
    while true
        owner = Base.@atomic order.owner
        previous = owner === nothing ? nothing : owner.value
        previous === task && return
        if previous === nothing || istaskdone(previous)
            _, bound = Base.@atomicreplace order.owner owner => WeakRef(task)
            bound && return
        else
            Base.@atomic order.shared = true
            return
        end
    end
end

mark_completed!(order::StreamOrder, epoch::UInt64) = (Base.@atomic order.completed max epoch; return)
is_completed(order::StreamOrder, epoch::UInt64) = epoch <= Base.@atomic order.completed

# record that work submitted to `order` from now on runs after `source`'s accesses up to `epoch`
function mark_ordered!(order::StreamOrder, source::StreamOrder, epoch::UInt64)
    Base.@lock order.waited begin
        order.waited[source] = max(get(order.waited, source, UInt64(0)), epoch)
    end
    return
end

# whether work submitted to `order` now runs after `source`'s accesses up to `epoch`
function is_ordered(order::StreamOrder, source::StreamOrder, epoch::UInt64)
    Base.@lock order.waited begin
        epoch <= get(order.waited, source, UInt64(0))
    end
end

"""
    CuStream(; flags=STREAM_DEFAULT, priority=nothing)

Create a CUDA stream.
"""
mutable struct CuStream
    const handle::CUstream
    Base.@atomic valid::Bool

    const ctx::Union{Nothing,CuContext}

    # when destroyed by the garbage collector, an event recorded right before, which
    # captures the work that was submitted to the stream (a `CuEvent`, if any)
    final_event::Any

    # bumped when the stream is handed to another task (see `create_stream`), which only
    # happens when it is idle, so work submitted during earlier generations has finished.
    Base.@atomic generation::Int

    # the special streams don't have one, as they implicitly synchronize with other streams
    const order::Union{Nothing,StreamOrder}

    function CuStream(; flags::CUstream_flags=STREAM_DEFAULT,
                        priority::Union{Nothing,Integer}=nothing)
        handle_ref = Ref{CUstream}()
        if priority === nothing
            cuStreamCreate(handle_ref, flags)
        else
            priority in priority_range() || throw(ArgumentError("Priority is out of range"))
            cuStreamCreateWithPriority(handle_ref, flags, priority)
        end

        ctx = current_context()
        # make sure memory last used on this stream can be released after destroying it
        disposal_stream(ctx)
        obj = new(handle_ref[], true, ctx, nothing, 0, StreamOrder())
        # destroying a stream from a finalizer is deferred, as with all resources
        # (see `retire!`), also so that memory that was last used on it can be freed first
        resource_finalizer(obj)
        return obj
    end

    global default_stream() = new(convert(CUstream, C_NULL), true, nothing, nothing, 0, nothing)

    global legacy_stream() = new(convert(CUstream, 1), true, nothing, nothing, 0, nothing)

    global per_thread_stream() = new(convert(CUstream, 2), true, nothing, nothing, 0, nothing)
end

"""
    default_stream()

Return the default stream.

!!! note

    It is generally better to use `stream()` to get a stream object that's local to the
    current task. That way, operations scheduled in other tasks can overlap.
"""
default_stream()

"""
    legacy_stream()

Return a special object to use use an implicit stream with legacy synchronization behavior.

You can use this stream to perform operations that should block on all streams (with the
exception of streams created with `STREAM_NON_BLOCKING`). This matches the old pre-CUDA 7
global stream behavior.
"""
legacy_stream()

"""
    per_thread_stream()

Return a special object to use an implicit stream with per-thread synchronization behavior.
This stream object is normally meant to be used with APIs that do not have per-thread
versions of their APIs (i.e. without a `ptsz` or `ptds` suffix).

!!! note

    It is generally not needed to use this type of stream. With CUDA.jl, each task already
    gets its own non-blocking stream, and multithreading in Julia is typically
    accomplished using tasks.
"""
per_thread_stream()

Base.unsafe_convert(::Type{CUstream}, s::CuStream) = s.handle

bind!(s::CuStream) = (s.order === nothing || bind!(s.order); return)

# only bumped while holding `stream_pool_lock` and `stream_disposal_lock`, but read without them
generation(s::CuStream) = Base.@atomic :acquire s.generation

Base.:(==)(a::CuStream, b::CuStream) = a.handle == b.handle
Base.hash(s::CuStream, h::UInt) = hash(s.handle, h)

@enum_without_prefix visibility=:public CUstream_flags_enum CU_

function unsafe_destroy!(s::CuStream)
    @assert s.ctx !== nothing "Cannot destroy unassociated stream"
    context!(s.ctx) do
        cuStreamDestroy_v2(s)
    end
    Base.@atomic s.valid = false
end

function Base.show(io::IO, stream::CuStream)
    print(io, "CuStream(")
    @printf(io, "%p", stream.handle)
    if stream.ctx !== nothing
        print(io, ", ", stream.ctx)
    end
    print(io, ")")
end

function unique_id(s::CuStream)
    id_ref = Ref{Culonglong}()
    cuStreamGetId(s, id_ref)
    return id_ref[]
end

"""
    isvalid(s::CuStream)

Determines if the stream object is still valid, i.e., if it has not been garbage collected.
This is only useful for use in finalizers, which do not guarantee order of execution (i.e.,
a stream may have been destroyed before an object relying on it has).
"""

function isvalid(s::CuStream)
    return s.valid
end

"""
    isdone(s::CuStream)

Return `false` if a stream is busy (has task running or queued)
and `true` if that stream is free.
"""
function isdone(s::CuStream)
    res = unchecked_cuStreamQuery(s)
    if res == ERROR_NOT_READY
        return false
    elseif res == SUCCESS
        return true
    else
        throw_api_error(res)
    end
end

"""
    synchronize([stream::CuStream])

Wait until `stream` has finished executing, with `stream` defaulting to the stream
associated with the current Julia task.

See also: [`device_synchronize`](@ref)
"""
synchronize(stream::CuStream=stream())

"""
    priority_range()

Return the valid range of stream priorities as a `StepRange` (with step size  1). The lower
bound of the range denotes the least priority (typically 0), with the upper bound
representing the greatest possible priority (typically -1).
"""
function priority_range()
    least_ref = Ref{Cint}()
    greatest_ref = Ref{Cint}()
    cuCtxGetStreamPriorityRange(least_ref, greatest_ref)
    step = least_ref[] < greatest_ref[] ? 1 : -1
    return least_ref[]:Cint(step):greatest_ref[]
end


"""
    priority(s::CuStream)

Return the priority of a stream `s`.
"""
function priority(s::CuStream)
    priority_ref = Ref{Cint}()
    cuStreamGetPriority(s, priority_ref)
    return priority_ref[]
end
