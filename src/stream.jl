# Stream management

import Base: unsafe_convert

export
    CuStream, default_stream, synchronize, destroy


typealias CuStream_t Ptr{Void}

immutable CuStream
    handle::CuStream_t
end

unsafe_convert(::Type{CuStream_t}, s::CuStream) = s.handle

function CuStream(flags::Integer)
    handle_ref = Ref{CuStream_t}()
    @apicall(:cuStreamCreate, (Ptr{CuStream_t}, Cuint),
                             handle_ref, flags)
    CuStream(handle_ref[])
end

CuStream() = CuStream(0)

destroy(s::CuStream) = @apicall(:cuStreamDestroy, (CuStream_t,), s.handle)

default_stream() = CuStream(convert(CuStream_t, C_NULL))

synchronize(s::CuStream) = @apicall(:cuStreamSynchronize, (CuStream_t,), s.handle)
