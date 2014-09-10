# CUDA CuContext

const CTX_SCHED_AUTO            = 0x00
const CTX_SCHED_SPIN            = 0x01
const CTX_SCHED_YIELD           = 0x02
const CTX_SCHED_BLOCKING_SYNC   = 0x04
const CTX_MAP_HOST              = 0x08
const CTX_LMEM_RESIZE_TO_MAX    = 0x10

immutable CuContext
	handle::Ptr{Void}

        function CuContext(dev::CuDevice, flags::Integer)
                a = Ptr{Void}[0]
                @cucall(:cuCtxCreate, (Ptr{Ptr{Void}}, Cuint, Cint), a, flags, dev.handle)
                new(a[1])
        end

        CuContext(dev::CuDevice) = CuContext(dev, 0)
end

function destroy(ctx::CuContext)
	@cucall(:cuCtxDestroy, (Ptr{Void},), ctx.handle)
end

function push(ctx::CuContext)
	@cucall(:cuCtxPushCurrent, (Ptr{Void},), ctx.handle)
end

function pop(ctx::CuContext)
	a = Ptr{Void}[0]
	@cucall(:cuCtxPopCurrent, (Ptr{Ptr{Void}},), a)
	return CuContext(a[1])
end
