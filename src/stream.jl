# Stream management

export
    CuStream, default_stream, synchronize, destroy


immutable CuStream
    handle::Ptr{Void}
    blocking::Bool
    priority::Int
end

default_stream() = CuStream(convert(Ptr{Void}, 0), true, 0)

synchronize(s::CuStream) = @cucall(:cuStreamSynchronize, (Ptr{Void},), s.handle)

destroy(s::CuStream) = @cucall(:cuStreamDestroy, (Ptr{Void},), s.handle)
