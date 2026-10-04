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
