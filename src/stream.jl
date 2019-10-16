# Stream management

export
    CuStream, CuDefaultStream, synchronize


mutable struct CuStream
    handle::CUstream
    ctx::CuContext
end

Base.unsafe_convert(::Type{CUstream}, s::CuStream) = s.handle

Base.:(==)(a::CuStream, b::CuStream) = a.handle == b.handle
Base.hash(s::CuStream, h::UInt) = hash(s.handle, h)

"""
    CuStream(flags=STREAM_DEFAULT)

Create a CUDA stream.
"""
function CuStream(flags::CUstream_flags=STREAM_DEFAULT)
    handle_ref = Ref{CUstream}()
    cuStreamCreate(handle_ref, flags)

    ctx = CuCurrentContext()
    obj = CuStream(handle_ref[], ctx)
    finalizer(unsafe_destroy!, obj)
    return obj
end

function unsafe_destroy!(s::CuStream)
    if isvalid(s.ctx)
        cuStreamDestroy(s)
    end
end

"""
    CuDefaultStream()

Return the default stream.
"""
@inline CuDefaultStream() = CuStream(convert(CUstream, C_NULL), CuContext(C_NULL))

"""
    synchronize(s::CuStream)

Wait until a stream's tasks are completed.
"""
synchronize(s::CuStream) = cuStreamSynchronize(s)
