# Stream management

export
    CuStream, CuDefaultStream, CuStreamLegacy, CuStreamPerThread,
    priority, priority_range, synchronize


mutable struct CuStream
    handle::CUstream
    ctx::CuContext
end

Base.unsafe_convert(::Type{CUstream}, s::CuStream) = s.handle

Base.:(==)(a::CuStream, b::CuStream) = a.handle == b.handle
Base.hash(s::CuStream, h::UInt) = hash(s.handle, h)

@enum_without_prefix CUstream_flags_enum CU_

"""
    CuStream(; flags=STREAM_DEFAULT, priority=nothing)

Create a CUDA stream.
"""
function CuStream(; flags::CUstream_flags=STREAM_DEFAULT,
                    priority::Union{Nothing,Integer}=nothing)
    handle_ref = Ref{CUstream}()
    if priority === nothing
        cuStreamCreate(handle_ref, flags)
    else
        priority in priority_range() || throw(ArgumentError("Priority is out of range"))
        cuStreamCreateWithPriority(handle_ref, flags, priority)
    end

    ctx = CuCurrentContext()
    obj = CuStream(handle_ref[], ctx)
    finalizer(unsafe_destroy!, obj)
    return obj
end

function unsafe_destroy!(s::CuStream)
    if isvalid(s.ctx)
        cuStreamDestroy_v2(s)
    end
end

"""
    CuDefaultStream()

Return the default stream.
"""
@inline CuDefaultStream() = CuStream(convert(CUstream, C_NULL), CuContext(C_NULL))

"""
    CuStreamLegacy()

Return a special object to use use an implicit stream with legacy synchronization behavior.

You can use this stream to perform operations that should block on all streams (with the
exception of streams created with `CU_STREAM_NON_BLOCKING`). This matches the old pre-CUDA 7
global stream behavior.
"""
@inline CuStreamLegacy() = CuStream(convert(CUstream, 1), CuContext(C_NULL))

"""
    CuStreamPerThread()

Return a special object to use an implicit stream with per-thread synchronization behavior.

This should generally only be used with compiled libraries, which cannot be switched to the
per-thread API calls. For all other uses, it be libraries compiled with `nvcc
--default-stream per-thread` or any CUDA API call using CUDA.jl (which defaults to the
per-thread variants) you can just use the default `CuDefaultStream` object.
"""
@inline CuStreamPerThread() = CuStream(convert(CUstream, 2), CuContext(C_NULL))

"""
    synchronize(s::CuStream)

Wait until a stream's tasks are completed.
"""
synchronize(s::CuStream) = cuStreamSynchronize(s)

"""
    query(s::CuStream)

Return `false` if a stream is busy (has task running or queued)
and `true` if that stream is free.
"""
function query(s::CuStream)
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
