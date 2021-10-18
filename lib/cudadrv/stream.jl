# Stream management

export
    CuStream, default_stream, legacy_stream, per_thread_stream,
    priority, priority_range, synchronize, device_synchronize

"""
    CuStream(; flags=STREAM_DEFAULT, priority=nothing)

Create a CUDA stream.
"""
mutable struct CuStream
    handle::CUstream
    ctx::Union{CuContext,Nothing}

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
        obj = new(handle_ref[], ctx)
        finalizer(unsafe_destroy!, obj)
        return obj
    end

    global default_stream() = new(convert(CUstream, C_NULL), nothing)

    global legacy_stream() = new(convert(CUstream, 1), nothing)

    global per_thread_stream() = new(convert(CUstream, 2), nothing)
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

Base.:(==)(a::CuStream, b::CuStream) = a.handle == b.handle
Base.hash(s::CuStream, h::UInt) = hash(s.handle, h)

@enum_without_prefix CUstream_flags_enum CU_

function unsafe_destroy!(s::CuStream)
    @finalize_in_ctx s.ctx cuStreamDestroy_v2(s)
end

function Base.show(io::IO, stream::CuStream)
    print(io, "CuStream(")
    @printf(io, "%p", stream.handle)
    print(io, ", ", stream.ctx, ")")
end

"""
    isdone(s::CuStream)

Return `false` if a stream is busy (has task running or queued)
and `true` if that stream is free.
"""
function isdone(s::CuStream)
    res = unsafe_cuStreamQuery(s)
    if res == ERROR_NOT_READY
        return false
    elseif res == SUCCESS
        return true
    else
        throw_api_error(res)
    end
end

"""
    synchronize([stream::CuStream]; blocking=true)

Wait until `stream` has finished executing, with `stream` defaulting to the stream
associated with the current Julia task. If `blocking` is true (the default), the active
task will block to conserve CPU time. If latency is important, set `blocking` to false.

See also: [`device_synchronize`](@ref)
"""
function synchronize(stream::CuStream=stream(); blocking::Bool=true)
    # fast path
    isdone(stream) && @goto(exit)

    # minimize latency of short operations by busy-waiting,
    # initially without even yielding to other tasks
    spins = 0
    while blocking || spins < 256
        if spins < 32
            ccall(:jl_cpu_pause, Cvoid, ())
            # Temporary solution before we have gc transition support in codegen.
            ccall(:jl_gc_safepoint, Cvoid, ())
        else
            yield()
        end
        isdone(stream) && @goto(exit)
        spins += 1
    end

    # minimize CPU usage of long-running kernels
    # by waiting for an event signalled by CUDA
    event = Threads.Event()
    launch(; stream) do
        notify(event)
    end
    Base.wait(event)

    @label(exit)
    check_exceptions()
end

"""
    device_synchronize()

Block for the current device's tasks to complete. This is a heavyweight operation, typically
you only need to call [`synchronize`](@ref) which only synchronizes the stream associated
with the current task.
"""
device_synchronize() = synchronize(legacy_stream())

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
    priority_range(s::CuStream)

Return the priority of a stream `s`.
"""
function priority(s::CuStream)
    priority_ref = Ref{Cint}()
    cuStreamGetPriority(s, priority_ref)
    return priority_ref[]
end
