# Profiler interface

export
    @profile, @cuprofile

@enum(CUoutput_mode, CSV            = 0x00,
                     KEY_VALUE_PAIR = 0x01)

"""
    @profile

`@profile <expression>` runs your expression, while activating the CUDA profiler.
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

start() = @apicall(:cuProfilerStart, (Ptr{Void},), C_NULL)
stop() = @apicall(:cuProfilerStop, (Ptr{Void},), C_NULL)

end
