"""
    @sync ex

Run expression `ex` and synchronize the GPU afterwards. This is a CPU-friendly
synchronization, i.e. it performs a blocking synchronization without increasing CPU load. As
such, this operation is preferred over implicit synchronization (e.g. when performing a
memory copy) for high-performance applications.

It is also useful for timing code that executes asynchronously.
"""
macro sync(ex)
    quote
        local e = CuEvent(EVENT_BLOCKING_SYNC | EVENT_DISABLE_TIMING)
        local ret = $(esc(ex))
        record(e)
        synchronize(e)
        ret
    end
end
