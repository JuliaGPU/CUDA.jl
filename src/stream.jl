# CUDA stream management

immutable CuStream
	handle::Ptr{Void}
	blocking::Bool
	priority::Int

	function CuStream()  # the null stream
		new(convert(Ptr{Void}, 0), true, 0)
	end

	function CuStream(;block::Bool=true)
		a = Array(Ptr{Void}, 1)
		flag = block ? 0 : 1
		@cucall(:cuStreamCreate, (Ptr{Ptr{Void}}, Cuint), a, flag)
		new(a[0], block, 0)
	end
end

function destroy(s::CuStream)
	@cucall(:cuStreamDestroy, (Ptr{Void},), s.handle)
end

function synchronize(s::CuStream)
	@cucall(:cuStreamSynchronize, (Ptr{Void},), s.handle)
end

