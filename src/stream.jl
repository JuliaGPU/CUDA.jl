# Stream management

export
    CuStream, CuDefaultStream, synchronize


typealias CuStream_t Ptr{Void}

type CuStream
    handle::CuStream_t
    ctx::CuContext
end

Base.unsafe_convert(::Type{CuStream_t}, s::CuStream) = s.handle
Base.:(==)(a::CuStream, b::CuStream) = a.handle == b.handle

function CuStream(flags::Integer=0)
    handle_ref = Ref{CuStream_t}()
    @apicall(:cuStreamCreate, (Ptr{CuStream_t}, Cuint),
                              handle_ref, flags)

    ctx = CuCurrentContext()
    obj = CuStream(handle_ref[], ctx)
    gc_track(ctx, obj)
    finalizer(obj, finalize)
    return obj
end

function finalize(s::CuStream)
    @apicall(:cuStreamDestroy, (CuModule_t,), s)
    gc_untrack(s.ctx, s)
end

CuDefaultStream() = CuStream(convert(CuStream_t, C_NULL), CuContext(C_NULL))

synchronize(s::CuStream) = @apicall(:cuStreamSynchronize, (CuStream_t,), s)
