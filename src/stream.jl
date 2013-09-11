# CUDA stream management

immutable CuStream
	handle::Ptr{Void}
	blocking::Bool
	priority::Int
end

function null_stream()
	CuStream(convert(Ptr{Void}, 0), true, 0)
end

function destroy(s::CuStream)
	@cucall(:cuStreamDestroy, (Ptr{Void},), s.handle)
end

function synchronize(s::CuStream)
	@cucall(:cuStreamSynchronize, (Ptr{Void},), s.handle)
end

