# Deprecated functionality

@deprecate CuDevice(ctx::CuContext) device(ctx)
@deprecate CuCurrentDevice() current_device()
@deprecate CuCurrentContext() current_context()
@deprecate CuContext(ptr::Union{Ptr,CuPtr}) context(ptr)
@deprecate CuDevice(ptr::Union{Ptr,CuPtr}) device(ptr)

@deprecate CuDefaultStream() default_stream()
@deprecate CuStreamLegacy() legacy_stream()
@deprecate CuStreamPerThread() per_thread_stream()
@deprecate query(s::CuStream) isdone(s)
@deprecate query(e::CuEvent) isdone(e)
