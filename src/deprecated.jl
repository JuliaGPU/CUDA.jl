# Deprecated functionality

export devcount, list_devices, destroy

@deprecate devcount() length(devices())
@deprecate list_devices() map(println, devices())
@deprecate destroy(ctx::CuContext) destroy!(ctx)
