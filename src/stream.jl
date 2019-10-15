# Stream management

export
    CuStream, CuDefaultStream, synchronize


const CuStream_t = Ptr{Cvoid}

mutable struct CuStream
    handle::CuStream_t
    ctx::CuContext
end

Base.unsafe_convert(::Type{CuStream_t}, s::CuStream) = s.handle

Base.:(==)(a::CuStream, b::CuStream) = a.handle == b.handle
Base.hash(s::CuStream, h::UInt) = hash(s.handle, h)

"""
    CuStream(flags=STREAM_DEFAULT)

Create a CUDA stream.
"""
function CuStream(flags::CUstream_flags=CU_STREAM_DEFAULT)
    handle_ref = Ref{CuStream_t}()
    @apicall(:cuStreamCreate, (Ptr{CuStream_t}, Cuint),
                              handle_ref, flags)

    ctx = CuCurrentContext()
    obj = CuStream(handle_ref[], ctx)
    finalizer(unsafe_destroy!, obj)
    return obj
end

function unsafe_destroy!(s::CuStream)
    if isvalid(s.ctx)
        @apicall(:cuStreamDestroy, (CuStream_t,), s)
    end
end

"""
    CuDefaultStream()

Return the default stream.
"""
@inline CuDefaultStream() = CuStream(convert(CuStream_t, C_NULL), CuContext(C_NULL))

"""
    synchronize(s::CuStream)

Wait until a stream's tasks are completed.
"""
synchronize(s::CuStream) = @apicall(:cuStreamSynchronize, (CuStream_t,), s)
