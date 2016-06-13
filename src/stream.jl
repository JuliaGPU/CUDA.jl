# Stream management

export
    CuStream, default_stream, synchronize, destroy


immutable CuStream
    handle::Ptr{Void}
end

function CuStream(flags::Integer)
    pctx_ref = Ref{Ptr{Void}}()
    @apicall(:cuStreamCreate, (Ptr{Ptr{Void}}, Cuint),
                             pctx_ref, flags)
    CuStream(pctx_ref[])
end

CuStream() = CuStream(0)

destroy(s::CuStream) = @apicall(:cuStreamDestroy, (Ptr{Void},), s.handle)

default_stream() = CuStream(convert(Ptr{Void}, 0))

synchronize(s::CuStream) = @apicall(:cuStreamSynchronize, (Ptr{Void},), s.handle)
