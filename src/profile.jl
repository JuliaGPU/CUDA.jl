# Profiler control

export
    @profile, @cuprofile

"""
    @profile ex

Run expressions while activating the CUDA profiler.

Note that this API is used to programmatically control the profiling granularity by allowing
profiling to be done only on selective pieces of code. It does not perform any profiling on
itself, you need external tools for that.
"""
macro profile(ex)
    quote
        Profile.start()
        local ret = $(esc(ex))
        Profile.stop()
        ret
    end
end


module Profile

using ..CUDAdrv


"""
    start()

Enables profile collection by the active profiling tool for the current context. If
profiling is already enabled, then this call has no effect.
"""
start() = CUDAdrv.cuProfilerStart()

"""
    stop()

Disables profile collection by the active profiling tool for the current context. If
profiling is already disabled, then this call has no effect.
"""
stop() = CUDAdrv.cuProfilerStop()

end
