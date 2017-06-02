# Profiler control

export
    @profile, @cuprofile

@enum(CUoutput_mode, CSV            = 0x00,
                     KEY_VALUE_PAIR = 0x01)

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
        try
            $(esc(ex))
        finally
            Profile.stop()
        end
    end
end


module Profile

using CUDAdrv
import CUDAdrv: @apicall

"""
    start()

Enables profile collection by the active profiling tool for the current context. If
profiling is already enabled, then this call has no effect.
"""
start() = @apicall(:cuProfilerStart, (Ptr{Void},), C_NULL)

"""
    stop()

Disables profile collection by the active profiling tool for the current context. If
profiling is already disabled, then this call has no effect.
"""
stop() = @apicall(:cuProfilerStop, (Ptr{Void},), C_NULL)

end
