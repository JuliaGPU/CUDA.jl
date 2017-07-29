# Deprecated functionality

import Base: @deprecate_binding

@deprecate devcount() length(devices())
@deprecate list_devices() map(println, devices())
@deprecate destroy(ctx::CuContext) destroy!(ctx)
@deprecate CuPrimaryContext(devnum::Int) CuPrimaryContext(CuDevice(devnum))
@deprecate reset(pctx::CuPrimaryContext) unsafe_reset!(pctx)
@deprecate_binding DevicePtr Ptr
