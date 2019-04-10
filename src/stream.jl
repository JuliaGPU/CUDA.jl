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

@enum(CUstream_flags, STREAM_DEFAULT      = 0x00,
                      STREAM_NON_BLOCKING = 0x01)

# FIXME: EnumSet from JuliaLang/julia#19470
Base.:|(x::CUstream_flags, y::CUstream_flags) =
    reinterpret(CUstream_flags, Base.cconvert(Unsigned, x) | Base.cconvert(Unsigned, y))

"""
    CuStream(flags=STREAM_DEFAULT)

Create a CUDA stream.
"""
function CuStream(flags::CUstream_flags=STREAM_DEFAULT)
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
