# Stream management

export
    CuStream, default_stream, synchronize, destroy


immutable CuStream
    handle::Ptr{Void}
end

function CuStream(flags::Integer)
    pctx_ref = Ref{Ptr{Void}}()
    @cucall(:cuStreamCreate, (Ptr{Ptr{Void}}, Cuint),
                             pctx_ref, flags)
    CuStream(pctx_ref[])
end

CuStream() = CuStream(0)

destroy(s::CuStream) = @cucall(:cuStreamDestroy, (Ptr{Void},), s.handle)

default_stream() = CuStream(convert(Ptr{Void}, 0))

synchronize(s::CuStream) = @cucall(:cuStreamSynchronize, (Ptr{Void},), s.handle)
