# CUDA Context

immutable Context
	handle::Ptr{Void}
end

const CTX_SCHED_AUTO  = 0x00
const CTX_SCHED_SPIN  = 0x01
const CTX_SCHED_YIELD = 0x02
const CTX_SCHED_BLOCKING_SYNC = 0x04
const CTX_MAP_HOST = 0x08
const CTX_LMEM_RESIZE_TO_MAX = 0x10

function create_context(dev::Device, flags::Integer)
	a = Array(Ptr{Void}, 1)
	@cucall(:cuCtxCreate, (Ptr{Ptr{Void}}, Cuint, Cint), a, flags, dev.handle)
	return Context(a[1])
end

create_context(dev::Device) = create_context(dev, 0)

function destroy(ctx::Context)
	@cucall(:cuCtxDestroy, (Ptr{Void},), ctx.handle)
end

function push(ctx::Context)
	@cucall(:cuCtxPushCurrent, (Ptr{Void},), ctx.handle)
end

function pop(ctx::Context)
	a = Array(Ptr{Void}, 1)
	@cucall(:cuCtxPopCurrent, (Ptr{Ptr{Void}},), a)
	return Context(a[1])
end

